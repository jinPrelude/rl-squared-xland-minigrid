import flax.nnx as nnx
import jax
import jax.numpy as jnp

from models.base import RL2Model
from models.obs_encoder import ObsEncoder


class RL2LSTM(RL2Model):
    def __init__(
        self,
        num_tiles: int,
        num_colors: int,
        obs_emb_dim: int,
        action_emb_dim: int,
        num_actions: int,
        hidden_dim: int = 256,
        num_steps: int | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        # encoded_dim: 5*5*2*obs_emb_dim + action_emb_dim(dir) + action_emb_dim(act) + 1(reward) + 1(done)
        encoded_dim = 5 * 5 * 2 * obs_emb_dim + 2 * action_emb_dim + 2

        self.encoder = ObsEncoder(
            num_tiles, num_colors, obs_emb_dim, action_emb_dim, num_actions, rngs=rngs,
        )
        self.fc1 = nnx.Linear(encoded_dim, hidden_dim, rngs=rngs)
        self.lstm = nnx.LSTMCell(hidden_dim, hidden_dim, rngs=rngs)
        self.fc_pi = nnx.Linear(hidden_dim, num_actions, rngs=rngs)
        self.fc_v = nnx.Linear(hidden_dim, 1, rngs=rngs)
        self.num_actions = num_actions
        self.num_steps = num_steps  # TBPTT chunk size (None = no truncation)

    def init_carry(self, batch_size: int, rngs: nnx.Rngs):
        return self.lstm.initialize_carry((batch_size, self.lstm.in_features), rngs=rngs)

    def step(self, obs_img, obs_dir, prev_action, prev_reward, done, carry):
        encoded = self.encoder(obs_img, obs_dir, prev_action, prev_reward, done)
        x = nnx.relu(self.fc1(encoded))
        carry, hidden = self.lstm(carry, x)
        logits = self.fc_pi(hidden)
        value = self.fc_v(hidden)
        return logits, value, carry

    def unroll(self, obs_img_seq, obs_dir_seq, prev_action_seq, prev_reward_seq, done_seq, init_carry):
        encoded_seq = self.encoder(obs_img_seq, obs_dir_seq, prev_action_seq, prev_reward_seq, done_seq)

        def scan_step(carry, x_t):
            x_t = nnx.relu(self.fc1(x_t))
            carry, hidden = self.lstm(carry, x_t)
            logits = self.fc_pi(hidden)
            value = self.fc_v(hidden)
            return carry, (logits, value)

        final_carry, (logits, values) = jax.lax.scan(scan_step, init_carry, encoded_seq)
        return logits, values, final_carry

    def prepare_training_data(self, transitions, advantages, returns, initial_carry):
        """Chunk full-trial rollout into TBPTT segments.

        If num_steps is None, falls back to base class (full trajectory).
        Otherwise reshapes (trial_length, num_envs, ...) -> (num_steps, num_chunks * num_envs, ...)
        and computes stop-gradient LSTM carries at each chunk boundary.
        """
        if self.num_steps is None:
            return super().prepare_training_data(transitions, advantages, returns, initial_carry)

        num_steps = self.num_steps
        trial_length = transitions.obs_img.shape[0]
        num_envs = transitions.obs_img.shape[1]
        num_chunks = trial_length // num_steps

        # 1. Compute LSTM carries at all timesteps via forward pass
        def _carry_step(carry, inputs):
            obs_img_t, obs_dir_t, prev_action_t, prev_reward_t, done_t = inputs
            _, _, new_carry = self.step(
                obs_img_t, obs_dir_t, prev_action_t, prev_reward_t, done_t, carry,
            )
            return new_carry, new_carry

        scan_inputs = (
            transitions.obs_img, transitions.obs_dir,
            transitions.prev_action, transitions.prev_reward, transitions.done,
        )
        _, all_carries = jax.lax.scan(_carry_step, initial_carry, scan_inputs)
        # all_carries: tree of (trial_length, num_envs, hidden_dim)

        # 2. Extract carries at chunk boundaries
        # Prepend initial_carry so index i gives the input carry for timestep i
        all_input_carries = jax.tree.map(
            lambda init, seq: jnp.concatenate([init[None], seq], axis=0),
            initial_carry, all_carries,
        )
        chunk_indices = jnp.arange(num_chunks) * num_steps
        chunk_carries = jax.tree.map(lambda c: c[chunk_indices], all_input_carries)
        chunk_carries = jax.lax.stop_gradient(chunk_carries)

        # 3. Reshape: (trial_length, num_envs, ...) -> (num_steps, num_chunks * num_envs, ...)
        def _reshape(arr):
            chunked = arr.reshape(num_chunks, num_steps, *arr.shape[1:])
            reordered = jnp.swapaxes(chunked, 0, 1)
            return reordered.reshape(num_steps, num_chunks * num_envs, *arr.shape[2:])

        merged_transitions = jax.tree.map(_reshape, transitions)
        merged_advantages = _reshape(advantages)
        merged_returns = _reshape(returns)
        merged_carries = jax.tree.map(
            lambda c: c.reshape(num_chunks * num_envs, *c.shape[2:]),
            chunk_carries,
        )

        effective_num_envs = num_chunks * num_envs
        return merged_transitions, merged_advantages, merged_returns, merged_carries, effective_num_envs
