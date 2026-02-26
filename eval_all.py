"""Batch evaluation on the test benchmark with CSV output."""

import csv
from argparse import ArgumentParser

import flax.nnx as nnx
import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    parser.add_argument("--num-steps-per-env", type=int, default=2560)
    parser.add_argument("--eval-num-envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="eval_results.csv")
    parser.add_argument("--num-segments", type=int, default=10)
    parser.add_argument("--plot-output", type=str, default="eval_segment_rewards.png")
    return parser.parse_args()


def plot_segment_rewards(step_rewards, num_segments, save_path, num_envs):
    """Plot per-segment reward sums averaged across environments."""
    num_steps = step_rewards.shape[1]
    segment_size = num_steps // num_segments
    segmented = step_rewards[:, :segment_size * num_segments].reshape(
        num_envs, num_segments, segment_size,
    )
    segment_sums = segmented.sum(axis=2)             # (num_envs, num_segments)
    means = segment_sums.mean(axis=0)                # (num_segments,)
    stds = segment_sums.std(axis=0)                  # (num_segments,)

    x = np.arange(1, num_segments + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(x, means - stds, means + stds, alpha=0.2, color="steelblue")
    ax.plot(x, means, marker="o", markersize=4, color="steelblue")
    ax.set_xlabel("Segment")
    ax.set_ylabel("Reward sum (mean across envs)")
    ax.set_title(f"Segment-wise Reward Mean ({num_segments} segments, {num_envs} envs)")
    ax.set_xticks(x)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


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

    print(f"Evaluating on {args.eval_num_envs} test environments ({args.num_steps_per_env} steps each)...")
    eval_rng_keys = jax.random.split(run_rng, args.eval_num_envs)
    eval_stats, step_rewards = jax.vmap(
        lambda rng_k, params: evaluate(model, env, params, rng_k, args.num_steps_per_env),
        in_axes=(0, 0),
    )(eval_rng_keys, eval_env_params)

    reward_sums = np.asarray(eval_stats.reward_sum)
    length_sums = np.asarray(eval_stats.length_sum)
    episodes_done = np.asarray(eval_stats.episodes_done)
    step_rewards = np.asarray(step_rewards)  # (num_envs, num_steps)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["env_index", "reward_sum", "length_sum", "episodes_done"])
        for i in range(args.eval_num_envs):
            writer.writerow([i, reward_sums[i], length_sums[i], episodes_done[i]])
    print(f"Saved results to {args.output}")

    print(f"\n--- Summary ({args.eval_num_envs} environments) ---")
    print(f"Reward  mean: {reward_sums.mean():.4f}  median: {np.median(reward_sums):.4f}  std: {reward_sums.std():.4f}")
    print(f"Length  mean: {length_sums.mean():.1f}")

    plot_segment_rewards(step_rewards, args.num_segments, args.plot_output, args.eval_num_envs)
    print(f"Saved segment reward plot to {args.plot_output}")


if __name__ == "__main__":
    main()
