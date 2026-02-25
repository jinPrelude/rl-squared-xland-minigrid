"""
eval_all.py - Batch evaluation on the test benchmark with CSV output.

Loads a trained RL² checkpoint, evaluates on multiple test environments
in parallel using jax.vmap, and writes per-environment results to CSV.

Usage:
    python eval_all.py --ckpt-dir checkpoints/rl2_gru_XLand-MiniGrid-R1-9x9_20260225_160340
"""

import csv
from argparse import ArgumentParser
from pathlib import Path

import flax.nnx as nnx
import jax
from jax import numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import xminigrid
from xminigrid.wrappers import GymAutoResetWrapper, DirectionObservationWrapper

from algorithms.rl_squared import evaluate
from models.lstm import RL2LSTM
from models.gru import RL2GRU

NUM_TILES = 13
NUM_COLORS = 12

MODEL_REGISTRY = {
    "lstm": RL2LSTM,
    "gru": RL2GRU,
}


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str, required=True,
                        help="Path to checkpoint run dir")
    parser.add_argument("--ckpt-step", type=int, default=None,
                        help="Step number to load. If None, uses latest.")
    parser.add_argument("--model", type=str, default="gru", choices=["lstm", "gru"])
    parser.add_argument("--env-id", type=str, default="XLand-MiniGrid-R1-9x9")
    parser.add_argument("--benchmark-id", type=str, default="small-1m")
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--rnn-hidden-dim", type=int, default=1024)
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--obs-emb-dim", type=int, default=16)
    parser.add_argument("--action-emb-dim", type=int, default=16)
    parser.add_argument("--episodes-per-trial", type=int, default=15)
    parser.add_argument("--eval-num-envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0,
                        help="Benchmark shuffle seed (must match training)")
    parser.add_argument("--eval-seed", type=int, default=42,
                        help="RNG seed for evaluation ruleset sampling and rollouts")
    parser.add_argument("--output", type=str, default="eval_results.csv")
    return parser.parse_args()


def main():
    args = parse_arguments()

    # --- Environment ---
    env, env_params = xminigrid.make(args.env_id)
    env = GymAutoResetWrapper(env)
    env = DirectionObservationWrapper(env)
    num_actions = env.num_actions(env_params)

    # --- Model ---
    rngs = nnx.Rngs(0)
    model = MODEL_REGISTRY[args.model](
        num_tiles=NUM_TILES,
        num_colors=NUM_COLORS,
        obs_emb_dim=args.obs_emb_dim,
        action_emb_dim=args.action_emb_dim,
        num_actions=num_actions,
        rnn_hidden_dim=args.rnn_hidden_dim,
        head_hidden_dim=args.head_hidden_dim,
        rngs=rngs,
    )

    # --- Load checkpoint ---
    ckpt_dir = Path(args.ckpt_dir).resolve()
    ckpt_mngr = ocp.CheckpointManager(ckpt_dir)
    step = args.ckpt_step if args.ckpt_step is not None else ckpt_mngr.latest_step()
    print(f"Loading checkpoint step {step} from {ckpt_dir}")
    _, state = nnx.split(model)
    restored_state = ckpt_mngr.restore(step, args=ocp.args.StandardRestore(state))
    nnx.update(model, restored_state)

    # --- Test benchmark (reproduce main.py split) ---
    benchmark = xminigrid.load_benchmark(args.benchmark_id)
    benchmark = benchmark.shuffle(jax.random.key(args.seed))
    _, test_benchmark = benchmark.split(args.train_split)

    # --- Sample eval rulesets ---
    eval_rng = jax.random.key(args.eval_seed)
    eval_rng, ruleset_rng, run_rng = jax.random.split(eval_rng, 3)

    eval_ruleset_keys = jax.random.split(ruleset_rng, args.eval_num_envs)
    eval_rulesets = jax.vmap(test_benchmark.sample_ruleset)(eval_ruleset_keys)
    eval_env_params = env_params.replace(ruleset=eval_rulesets)

    # --- Batch evaluation ---
    print(f"Evaluating on {args.eval_num_envs} test environments "
          f"({args.episodes_per_trial} episodes each)...")

    eval_rng_keys = jax.random.split(run_rng, args.eval_num_envs)

    eval_stats = jax.vmap(
        lambda rng_k, params: evaluate(
            model, env, params, rng_k, args.episodes_per_trial
        ),
        in_axes=(0, 0),
    )(eval_rng_keys, eval_env_params)

    reward_sums = np.asarray(eval_stats.reward_sum)
    length_sums = np.asarray(eval_stats.length_sum)
    episodes_done = np.asarray(eval_stats.episodes_done)

    # --- Write CSV ---
    output_path = Path(args.output)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["env_index", "reward_sum", "length_sum", "episodes_done"])
        for i in range(args.eval_num_envs):
            writer.writerow([i, reward_sums[i], length_sums[i], episodes_done[i]])

    print(f"Saved per-environment results to {output_path}")

    # --- Summary ---
    print(f"\n--- Summary ({args.eval_num_envs} environments) ---")
    print(f"Reward  mean:   {reward_sums.mean():.4f}")
    print(f"Reward  median: {np.median(reward_sums):.4f}")
    print(f"Reward  std:    {reward_sums.std():.4f}")
    print(f"Length  mean:   {length_sums.mean():.1f}")


if __name__ == "__main__":
    main()
