from argparse import ArgumentParser

import flax.nnx as nnx
import jax
from jax import numpy as jnp
import numpy as np
import optax
import wandb
import xminigrid
from xminigrid.wrappers import GymAutoResetWrapper, DirectionObservationWrapper

from models.lstm import RL2LSTM
from algorithms.rl_squared import (
    rollout_scan,
    bootstrap_value,
    calculate_gae,
    make_minibatches,
    update_ppo,
    evaluate,
)


NUM_TILES = 13   # from xminigrid.core.constants
NUM_COLORS = 12  # from xminigrid.core.constants


def parse_arguments():
    parser = ArgumentParser()
    # Environment
    parser.add_argument("--env-id", type=str, default="XLand-MiniGrid-R1-9x9")
    parser.add_argument("--benchmark-id", type=str, default="small-1m")
    parser.add_argument("--train-split", type=float, default=0.8)
    # Seeds
    parser.add_argument("--seed", type=int, default=0)
    # Training
    parser.add_argument("--num-iter", type=int, default=10000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episodes-per-trial", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--obs-emb-dim", type=int, default=16)
    parser.add_argument("--action-emb-dim", type=int, default=16)
    parser.add_argument("--envs-per-batch", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lmbda", type=float, default=0.97)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    # Evaluation
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-num-envs", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_arguments()
    assert args.num_envs % args.envs_per_batch == 0

    # --- Environment setup ---
    env, env_params = xminigrid.make(args.env_id)
    env = GymAutoResetWrapper(env)
    env = DirectionObservationWrapper(env)

    num_actions = env.num_actions(env_params)
    max_episode_steps = env_params.max_steps  # e.g., 243 for 9x9
    trial_length = args.episodes_per_trial * max_episode_steps

    # --- Benchmark: load and split into train/test ---
    benchmark = xminigrid.load_benchmark(args.benchmark_id)
    rng = jax.random.key(args.seed)
    rng, shuffle_rng = jax.random.split(rng)
    benchmark = benchmark.shuffle(shuffle_rng)
    train_benchmark, test_benchmark = benchmark.split(args.train_split)

    # --- W&B ---
    wandb.init(project="rl-squared-xland", name=f"rl2_{args.env_id}", config=vars(args))

    # --- Model & optimizer ---
    rngs = nnx.Rngs(args.seed)
    model = RL2LSTM(
        num_tiles=NUM_TILES,
        num_colors=NUM_COLORS,
        obs_emb_dim=args.obs_emb_dim,
        action_emb_dim=args.action_emb_dim,
        num_actions=num_actions,
        hidden_dim=args.hidden_dim,
        rngs=rngs,
    )
    optimizer = nnx.Optimizer(model, optax.adamw(args.learning_rate), wrt=nnx.Param)

    metrics = nnx.metrics.MultiMetric(
        critic_loss=nnx.metrics.Average("critic_loss"),
        actor_loss=nnx.metrics.Average("actor_loss"),
    )

    global_env_step = 0

    for iteration in range(args.num_iter):
        # === Meta-episode setup ===
        rng, ruleset_rng, reset_rng, rollout_rng = jax.random.split(rng, 4)

        # Sample rulesets from TRAIN benchmark
        ruleset_keys = jax.random.split(ruleset_rng, args.num_envs)
        rulesets = jax.vmap(train_benchmark.sample_ruleset)(ruleset_keys)
        meta_env_params = env_params.replace(ruleset=rulesets)

        # Reset all envs (vmapped)
        reset_keys = jax.random.split(reset_rng, args.num_envs)
        timestep = jax.vmap(env.reset, in_axes=(0, 0))(meta_env_params, reset_keys)

        # Initialize LSTM carry
        carry = model.init_carry(args.num_envs, rngs)
        initial_carry = jax.tree.map(lambda c: c.copy(), carry)
        prev_action = jnp.zeros(args.num_envs, dtype=jnp.int32)
        prev_reward = jnp.zeros(args.num_envs)

        # === Scan-based rollout ===
        (final_timestep, final_prev_action, final_prev_reward, final_carry, _), transitions = \
            rollout_scan(model, env, meta_env_params, timestep, prev_action, prev_reward,
                         carry, rollout_rng, trial_length)

        global_env_step += trial_length * args.num_envs

        # === GAE (meta-RL: no episode boundary masking) ===
        next_value = bootstrap_value(model, final_timestep, final_prev_action,
                                     final_prev_reward, final_carry)
        advantages, returns = calculate_gae(transitions, next_value,
                                            gamma=args.gamma, lmbda=args.lmbda)

        # === PPO update ===
        num_minibatches = args.num_envs // args.envs_per_batch
        for _ in range(args.num_epochs):
            rng, perm_rng = jax.random.split(rng)
            env_indices = jax.random.permutation(perm_rng, args.num_envs)
            env_indices = env_indices[:num_minibatches * args.envs_per_batch]
            minibatches = make_minibatches(transitions, advantages, returns,
                                           initial_carry, env_indices, args.envs_per_batch)
            update_ppo(model, optimizer, minibatches, metrics, clip_eps=args.clip_eps)

        # === Logging ===
        metric_values = {k: float(v) for k, v in metrics.compute().items()}
        wandb_payload = {
            "train/iteration": iteration,
            "train/global_env_step": global_env_step,
            "train/actor_loss": metric_values["actor_loss"],
            "train/critic_loss": metric_values["critic_loss"],
            "train/reward_mean": float(transitions.reward.sum(axis=0).mean()),
        }

        # === Evaluation on test benchmark (every eval_interval iterations) ===
        if (iteration + 1) % args.eval_interval == 0:
            rng, eval_ruleset_rng, eval_run_rng = jax.random.split(rng, 3)

            eval_ruleset_keys = jax.random.split(eval_ruleset_rng, args.eval_num_envs)
            eval_rulesets = jax.vmap(test_benchmark.sample_ruleset)(eval_ruleset_keys)
            eval_env_params = env_params.replace(ruleset=eval_rulesets)

            eval_rng_keys = jax.random.split(eval_run_rng, args.eval_num_envs)

            eval_stats = jax.vmap(
                lambda rng_k, params: evaluate(model, env, params, rng_k, args.episodes_per_trial),
                in_axes=(0, 0),
            )(eval_rng_keys, eval_env_params)

            wandb_payload["test/reward_mean"] = float(eval_stats.reward_sum.mean())
            wandb_payload["test/reward_median"] = float(jnp.median(eval_stats.reward_sum))
            wandb_payload["test/length_mean"] = float(eval_stats.length_sum.mean())

        wandb.log(wandb_payload, step=global_env_step)
        metrics.reset()


if __name__ == "__main__":
    main()
