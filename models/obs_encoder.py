import flax.nnx as nnx
import jax.numpy as jnp


class ObsEncoder(nnx.Module):
    """Encodes XLand-MiniGrid structured observations into a flat vector."""

    def __init__(
        self,
        num_tiles: int,
        num_colors: int,
        obs_emb_dim: int,
        action_emb_dim: int,
        num_actions: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.tile_emb = nnx.Embed(num_embeddings=num_tiles, features=obs_emb_dim, rngs=rngs)
        self.color_emb = nnx.Embed(num_embeddings=num_colors, features=obs_emb_dim, rngs=rngs)
        self.action_emb = nnx.Embed(num_embeddings=num_actions, features=action_emb_dim, rngs=rngs)
        self.dir_proj = nnx.Linear(4, action_emb_dim, rngs=rngs)
        self.obs_emb_dim = obs_emb_dim

    def __call__(self, obs_img, obs_dir, prev_action, prev_reward, done):
        # obs_img: (..., 5, 5, 2) integer grid [tile_type, color]
        # obs_dir: (..., 4) one-hot direction
        # prev_action: (...,) integer
        # prev_reward: (...,) float
        # done: (...,) float
        tile_enc = self.tile_emb(obs_img[..., 0].astype(jnp.int32))   # (..., 5, 5, obs_emb_dim)
        color_enc = self.color_emb(obs_img[..., 1].astype(jnp.int32)) # (..., 5, 5, obs_emb_dim)
        img_enc = jnp.concatenate([tile_enc, color_enc], axis=-1)      # (..., 5, 5, 2*obs_emb_dim)
        img_flat = img_enc.reshape(*obs_img.shape[:-3], -1)            # (..., 5*5*2*obs_emb_dim)

        dir_enc = self.dir_proj(obs_dir)              # (..., action_emb_dim)
        act_enc = self.action_emb(prev_action)        # (..., action_emb_dim)

        return jnp.concatenate([
            img_flat,
            dir_enc,
            act_enc,
            prev_reward[..., None],
            done[..., None],
        ], axis=-1)
