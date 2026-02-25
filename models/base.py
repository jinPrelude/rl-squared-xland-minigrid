import abc

import flax.nnx as nnx
import jax
import jax.numpy as jnp


class RL2Model(nnx.Module, abc.ABC):
    """Base class for RL² policy models."""

    @abc.abstractmethod
    def init_carry(self, batch_size: int, rngs: nnx.Rngs):
        """Initialize recurrent state."""
        ...

    @abc.abstractmethod
    def step(self, obs_img, obs_dir, prev_action, prev_reward, done, carry):
        """Single-step forward pass.

        Returns: (logits, value, new_carry)
        """
        ...

    @abc.abstractmethod
    def unroll(self, obs_img_seq, obs_dir_seq, prev_action_seq,
               prev_reward_seq, done_seq, init_carry):
        """Sequence forward pass.

        Returns: (logits, values, final_carry)
        """
        ...

    def prepare_training_data(self, transitions, advantages, returns, initial_carry):
        """Chunk full-trial rollout into TBPTT segments.

        If self.num_steps is None, passes through the full trajectory.
        Otherwise reshapes (trial_length, num_envs, ...) -> (num_steps, num_chunks * num_envs, ...)
        and computes stop-gradient carries at each chunk boundary.

        Returns: (transitions, advantages, returns, initial_carry, effective_num_envs)
        """
        if self.num_steps is None:
            num_envs = transitions.obs_img.shape[1]
            return transitions, advantages, returns, initial_carry, num_envs

        num_steps = self.num_steps
        trial_length = transitions.obs_img.shape[0]
        num_envs = transitions.obs_img.shape[1]
        num_chunks = trial_length // num_steps

        # 1. Compute carries at all timesteps via forward pass
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

        # 2. Extract carries at chunk boundaries
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
