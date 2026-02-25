import flax.nnx as nnx
import jax.numpy as jnp


class ObsEncoder(nnx.Module):
    """Encodes XLand-MiniGrid structured observations into a flat vector.

    Uses embedding + CNN for spatial compression of grid observations.
    """

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

        # CNN: (5,5, 2*obs_emb_dim) → (4,4,16) → (3,3,32) → (2,2,64)
        self.conv1 = nnx.Conv(2 * obs_emb_dim, 16, kernel_size=(2, 2), padding='VALID', rngs=rngs)
        self.conv2 = nnx.Conv(16, 32, kernel_size=(2, 2), padding='VALID', rngs=rngs)
        self.conv3 = nnx.Conv(32, 64, kernel_size=(2, 2), padding='VALID', rngs=rngs)

        self.action_emb = nnx.Embed(num_embeddings=num_actions, features=action_emb_dim, rngs=rngs)
        self.dir_proj = nnx.Linear(4, action_emb_dim, rngs=rngs)

    def __call__(self, obs_img, obs_dir, prev_action, prev_reward, done):
        # obs_img: (..., 5, 5, 2) integer grid [tile_type, color]
        # obs_dir: (..., 4) one-hot direction
        # prev_action: (...,) integer
        # prev_reward: (...,) float
        # done: (...,) float
        tile_enc = self.tile_emb(obs_img[..., 0].astype(jnp.int32))   # (..., 5, 5, obs_emb_dim)
        color_enc = self.color_emb(obs_img[..., 1].astype(jnp.int32)) # (..., 5, 5, obs_emb_dim)
        img_enc = jnp.concatenate([tile_enc, color_enc], axis=-1)      # (..., 5, 5, 2*obs_emb_dim)

        # CNN spatial compression
        x = nnx.relu(self.conv1(img_enc))   # (..., 4, 4, 16)
        x = nnx.relu(self.conv2(x))         # (..., 3, 3, 32)
        x = nnx.relu(self.conv3(x))         # (..., 2, 2, 64)
        img_flat = x.reshape(*obs_img.shape[:-3], -1)  # (..., 256)

        dir_enc = self.dir_proj(obs_dir)              # (..., action_emb_dim)
        act_enc = self.action_emb(prev_action)        # (..., action_emb_dim)

        return jnp.concatenate([
            img_flat,
            dir_enc,
            act_enc,
            prev_reward[..., None],
            done[..., None],
        ], axis=-1)
