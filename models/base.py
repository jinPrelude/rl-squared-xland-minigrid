import abc

import flax.nnx as nnx


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
