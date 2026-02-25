import flax.nnx as nnx
import jax
import jax.numpy as jnp

from models.base import RL2Model
from models.obs_encoder import ObsEncoder


class RL2GRU(RL2Model):
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
        self.gru = nnx.GRUCell(hidden_dim, hidden_dim, rngs=rngs)
        self.fc_pi = nnx.Linear(hidden_dim, num_actions, rngs=rngs)
        self.fc_v = nnx.Linear(hidden_dim, 1, rngs=rngs)
        self.num_actions = num_actions
        self.num_steps = num_steps  # TBPTT chunk size (None = no truncation)

    def init_carry(self, batch_size: int, rngs: nnx.Rngs):
        return self.gru.initialize_carry((batch_size, self.gru.in_features), rngs=rngs)

    def step(self, obs_img, obs_dir, prev_action, prev_reward, done, carry):
        encoded = self.encoder(obs_img, obs_dir, prev_action, prev_reward, done)
        x = nnx.relu(self.fc1(encoded))
        carry, hidden = self.gru(carry, x)
        logits = self.fc_pi(hidden)
        value = self.fc_v(hidden)
        return logits, value, carry

    def unroll(self, obs_img_seq, obs_dir_seq, prev_action_seq, prev_reward_seq, done_seq, init_carry):
        encoded_seq = self.encoder(obs_img_seq, obs_dir_seq, prev_action_seq, prev_reward_seq, done_seq)

        def scan_step(carry, x_t):
            x_t = nnx.relu(self.fc1(x_t))
            carry, hidden = self.gru(carry, x_t)
            logits = self.fc_pi(hidden)
            value = self.fc_v(hidden)
            return carry, (logits, value)

        final_carry, (logits, values) = jax.lax.scan(scan_step, init_carry, encoded_seq)
        return logits, values, final_carry
