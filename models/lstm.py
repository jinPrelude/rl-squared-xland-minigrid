import flax.nnx as nnx
import jax
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


class RL2LSTM(nnx.Module):
    def __init__(
        self,
        num_tiles: int,
        num_colors: int,
        obs_emb_dim: int,
        action_emb_dim: int,
        num_actions: int,
        hidden_dim: int = 256,
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
