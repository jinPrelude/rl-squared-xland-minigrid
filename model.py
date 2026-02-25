import flax.nnx as nnx
import jax
import jax.numpy as jnp

RNN_CELLS = {"lstm": nnx.LSTMCell, "gru": nnx.GRUCell}


class ObsEncoder(nnx.Module):
    def __init__(self, num_tiles, num_colors, obs_emb_dim, action_emb_dim, num_actions, *, rngs):
        self.tile_emb = nnx.Embed(num_embeddings=num_tiles, features=obs_emb_dim, rngs=rngs)
        self.color_emb = nnx.Embed(num_embeddings=num_colors, features=obs_emb_dim, rngs=rngs)
        self.conv1 = nnx.Conv(2 * obs_emb_dim, 16, kernel_size=(2, 2), padding='VALID', rngs=rngs)
        self.conv2 = nnx.Conv(16, 32, kernel_size=(2, 2), padding='VALID', rngs=rngs)
        self.conv3 = nnx.Conv(32, 64, kernel_size=(2, 2), padding='VALID', rngs=rngs)
        self.action_emb = nnx.Embed(num_embeddings=num_actions, features=action_emb_dim, rngs=rngs)
        self.dir_proj = nnx.Linear(4, action_emb_dim, rngs=rngs)

    def __call__(self, obs_img, obs_dir, prev_action, prev_reward, done):
        tile_enc = self.tile_emb(obs_img[..., 0].astype(jnp.int32))
        color_enc = self.color_emb(obs_img[..., 1].astype(jnp.int32))
        img_enc = jnp.concatenate([tile_enc, color_enc], axis=-1)
        x = nnx.relu(self.conv1(img_enc))
        x = nnx.relu(self.conv2(x))
        x = nnx.relu(self.conv3(x))
        img_flat = x.reshape(*obs_img.shape[:-3], -1)
        return jnp.concatenate([
            img_flat, self.dir_proj(obs_dir), self.action_emb(prev_action),
            prev_reward[..., None], done[..., None],
        ], axis=-1)


class RL2Model(nnx.Module):
    def __init__(self, num_tiles, num_colors, obs_emb_dim, action_emb_dim, num_actions,
                 rnn_hidden_dim=1024, head_hidden_dim=256, rnn_type="lstm", *, rngs):
        encoded_dim = 256 + 2 * action_emb_dim + 2
        self.encoder = ObsEncoder(num_tiles, num_colors, obs_emb_dim, action_emb_dim, num_actions, rngs=rngs)
        self.fc1 = nnx.Linear(encoded_dim, rnn_hidden_dim, rngs=rngs)
        self.rnn = RNN_CELLS[rnn_type](rnn_hidden_dim, rnn_hidden_dim, rngs=rngs)
        self.head_mlp = nnx.Linear(rnn_hidden_dim, head_hidden_dim, rngs=rngs)
        self.fc_pi = nnx.Linear(head_hidden_dim, num_actions, rngs=rngs)
        self.fc_v = nnx.Linear(head_hidden_dim, 1, rngs=rngs)

    def init_carry(self, batch_size, rngs):
        return self.rnn.initialize_carry((batch_size, self.rnn.in_features), rngs=rngs)

    def step(self, obs_img, obs_dir, prev_action, prev_reward, done, carry):
        x = nnx.relu(self.fc1(self.encoder(obs_img, obs_dir, prev_action, prev_reward, done)))
        carry, hidden = self.rnn(carry, x)
        hidden = nnx.relu(self.head_mlp(hidden))
        return self.fc_pi(hidden), self.fc_v(hidden), carry

    def unroll(self, obs_img_seq, obs_dir_seq, prev_action_seq, prev_reward_seq, done_seq, init_carry):
        encoded_seq = self.encoder(obs_img_seq, obs_dir_seq, prev_action_seq, prev_reward_seq, done_seq)

        def scan_step(carry, x_t):
            x_t = nnx.relu(self.fc1(x_t))
            carry, hidden = self.rnn(carry, x_t)
            hidden = nnx.relu(self.head_mlp(hidden))
            return carry, (self.fc_pi(hidden), self.fc_v(hidden))

        final_carry, (logits, values) = jax.lax.scan(scan_step, init_carry, encoded_seq)
        return logits, values, final_carry
