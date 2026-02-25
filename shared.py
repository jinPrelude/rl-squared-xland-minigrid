from pathlib import Path

import flax.nnx as nnx
import jax
import orbax.checkpoint as ocp
import xminigrid
from xminigrid.wrappers import GymAutoResetWrapper, DirectionObservationWrapper

from model import RL2Model

NUM_TILES = 13
NUM_COLORS = 12


def make_env(env_id):
    env, env_params = xminigrid.make(env_id)
    env = GymAutoResetWrapper(env)
    env = DirectionObservationWrapper(env)
    return env, env_params


def make_model(args, num_actions, rngs):
    return RL2Model(
        num_tiles=NUM_TILES, num_colors=NUM_COLORS,
        obs_emb_dim=args.obs_emb_dim, action_emb_dim=args.action_emb_dim,
        num_actions=num_actions, rnn_hidden_dim=args.rnn_hidden_dim,
        head_hidden_dim=args.head_hidden_dim, rnn_type=args.model, rngs=rngs,
    )


def load_checkpoint(model, ckpt_dir, step=None):
    ckpt_dir = Path(ckpt_dir).resolve()
    ckpt_mngr = ocp.CheckpointManager(ckpt_dir)
    step = step if step is not None else ckpt_mngr.latest_step()
    _, state = nnx.split(model)
    restored = ckpt_mngr.restore(step, args=ocp.args.StandardRestore(state))
    nnx.update(model, restored)
    return step


def load_test_benchmark(benchmark_id, seed, train_split):
    benchmark = xminigrid.load_benchmark(benchmark_id)
    benchmark = benchmark.shuffle(jax.random.key(seed))
    _, test_benchmark = benchmark.split(train_split)
    return test_benchmark
