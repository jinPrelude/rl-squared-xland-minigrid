"""Batch evaluation on the test benchmark with CSV output."""

import csv
from argparse import ArgumentParser

import flax.nnx as nnx
import jax
import numpy as np

from shared import make_env, make_model, load_checkpoint, load_test_benchmark
from rl_squared import evaluate


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str, required=True)
    parser.add_argument("--ckpt-step", type=int, default=None)
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="eval_results.csv")
    return parser.parse_args()


def main():
    args = parse_arguments()

    env, env_params = make_env(args.env_id)
    num_actions = env.num_actions(env_params)

    model = make_model(args, num_actions, nnx.Rngs(0))
    step = load_checkpoint(model, args.ckpt_dir, args.ckpt_step)
    print(f"Loaded checkpoint step {step}")

    test_benchmark = load_test_benchmark(args.benchmark_id, args.seed, args.train_split)

    eval_rng = jax.random.key(args.eval_seed)
    eval_rng, ruleset_rng, run_rng = jax.random.split(eval_rng, 3)
    eval_ruleset_keys = jax.random.split(ruleset_rng, args.eval_num_envs)
    eval_rulesets = jax.vmap(test_benchmark.sample_ruleset)(eval_ruleset_keys)
    eval_env_params = env_params.replace(ruleset=eval_rulesets)

    print(f"Evaluating on {args.eval_num_envs} test environments ({args.episodes_per_trial} episodes each)...")
    eval_rng_keys = jax.random.split(run_rng, args.eval_num_envs)
    eval_stats = jax.vmap(
        lambda rng_k, params: evaluate(model, env, params, rng_k, args.episodes_per_trial),
        in_axes=(0, 0),
    )(eval_rng_keys, eval_env_params)

    reward_sums = np.asarray(eval_stats.reward_sum)
    length_sums = np.asarray(eval_stats.length_sum)
    episodes_done = np.asarray(eval_stats.episodes_done)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["env_index", "reward_sum", "length_sum", "episodes_done"])
        for i in range(args.eval_num_envs):
            writer.writerow([i, reward_sums[i], length_sums[i], episodes_done[i]])
    print(f"Saved results to {args.output}")

    print(f"\n--- Summary ({args.eval_num_envs} environments) ---")
    print(f"Reward  mean: {reward_sums.mean():.4f}  median: {np.median(reward_sums):.4f}  std: {reward_sums.std():.4f}")
    print(f"Length  mean: {length_sums.mean():.1f}")


if __name__ == "__main__":
    main()
