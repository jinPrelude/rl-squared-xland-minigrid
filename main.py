import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import flax.nnx as nnx
import jax
from jax import numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
import xminigrid
from xminigrid.wrappers import GymAutoResetWrapper, DirectionObservationWrapper

from models.lstm import RL2LSTM
from models.gru import RL2GRU
from algorithms.rl_squared import (
    rollout_scan,
    bootstrap_value,
    calculate_gae,
    make_minibatches,
    update_ppo,
    evaluate,
)

# Constants from xminigrid.core.constants for observation encoder initialization
NUM_TILES = 13
NUM_COLORS = 12

MODEL_REGISTRY = {
    "lstm": RL2LSTM,
    "gru": RL2GRU,
}


def parse_arguments():
    parser = ArgumentParser()
    # Environment
    parser.add_argument("--env-id", type=str, default="XLand-MiniGrid-R1-9x9")
    parser.add_argument("--benchmark-id", type=str, default="small-1m")
    parser.add_argument("--train-split", type=float, default=0.8)
    # Seeds
    parser.add_argument("--seed", type=int, default=0)
    # Training
    parser.add_argument("--total-transitions", type=int, default=10_000_000_000, # 10B
                        help="Total number of environment transitions to collect")
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--episodes-per-trial", type=int, default=15)
    parser.add_argument("--num-steps-per-update", type=int, default=243,
                        help="Steps per inner collect-update cycle (must divide trial_length)")
    parser.add_argument("--rnn-hidden-dim", type=int, default=1024)
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--obs-emb-dim", type=int, default=16)
    parser.add_argument("--action-emb-dim", type=int, default=16)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lmbda", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "gru"],
                        help="Recurrent model type: lstm or gru")
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    # Evaluation
    parser.add_argument("--eval-interval", type=int, default=100_000_000,
                        help="Evaluate every this many transitions")
    parser.add_argument("--eval-num-envs", type=int, default=4096)
    # Checkpoint
    parser.add_argument("--save-ckpt-dir", type=str, default="./checkpoints")
    return parser.parse_args()


def main():
    args = parse_arguments()
    assert args.num_envs % args.num_minibatches == 0
    envs_per_batch = args.num_envs // args.num_minibatches

    # --- Environment setup ---
    env, env_params = xminigrid.make(args.env_id)
    env = GymAutoResetWrapper(env)
    env = DirectionObservationWrapper(env)

    num_actions = env.num_actions(env_params)
    max_episode_steps = env_params.max_steps  # e.g., 243 for 9x9
    trial_length = args.episodes_per_trial * max_episode_steps

    # Interleaved collect-update: trial split into inner updates
    assert trial_length % args.num_steps_per_update == 0, \
        f"trial_length ({trial_length}) must be divisible by num_steps_per_update ({args.num_steps_per_update})"
    num_inner_updates = trial_length // args.num_steps_per_update

    # --- Benchmark: load and split into train/test ---
    benchmark = xminigrid.load_benchmark(args.benchmark_id)
    rng = jax.random.key(args.seed)
    rng, shuffle_rng = jax.random.split(rng)
    benchmark = benchmark.shuffle(shuffle_rng)
    train_benchmark, test_benchmark = benchmark.split(args.train_split)

    # --- Run name with datetime ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"rl2_{args.model}_{args.env_id}_{timestamp}"

    # --- W&B ---
    wandb.init(project="rl-squared-xland", name=run_name, config=vars(args))

    # --- Model & optimizer ---
    rngs = nnx.Rngs(args.seed)
    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(
        num_tiles=NUM_TILES,
        num_colors=NUM_COLORS,
        obs_emb_dim=args.obs_emb_dim,
        action_emb_dim=args.action_emb_dim,
        num_actions=num_actions,
        rnn_hidden_dim=args.rnn_hidden_dim,
        head_hidden_dim=args.head_hidden_dim,
        rngs=rngs,
    )
    optimizer = nnx.Optimizer(model, optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adam(args.learning_rate),
    ), wrt=nnx.Param)

    metrics = nnx.metrics.MultiMetric(
        critic_loss=nnx.metrics.Average("critic_loss"),
        actor_loss=nnx.metrics.Average("actor_loss"),
        entropy_loss=nnx.metrics.Average("entropy_loss"),
    )

    # --- Checkpoint manager ---
    ckpt_dir = (Path(args.save_ckpt_dir) / run_name).resolve()
    ckpt_mngr = ocp.CheckpointManager(ckpt_dir)

    # --- Fixed eval rulesets (sampled once for consistent evaluation) ---
    rng, eval_ruleset_rng = jax.random.split(rng)
    eval_ruleset_keys = jax.random.split(eval_ruleset_rng, args.eval_num_envs)
    eval_rulesets = jax.vmap(test_benchmark.sample_ruleset)(eval_ruleset_keys)
    eval_env_params = env_params.replace(ruleset=eval_rulesets)

    global_env_step = 0
    iteration = 0
    last_eval_step = 0

    transitions_per_iter = trial_length * args.num_envs

    while global_env_step < args.total_transitions:
        iter_start = time.time()

        # === Meta-episode setup ===
        rng, ruleset_rng, reset_rng, rollout_rng = jax.random.split(rng, 4)

        # Sample rulesets from TRAIN benchmark
        ruleset_keys = jax.random.split(ruleset_rng, args.num_envs)
        rulesets = jax.vmap(train_benchmark.sample_ruleset)(ruleset_keys)
        meta_env_params = env_params.replace(ruleset=rulesets)

        # Reset all envs (vmapped)
        reset_keys = jax.random.split(reset_rng, args.num_envs)
        timestep = jax.vmap(env.reset, in_axes=(0, 0))(meta_env_params, reset_keys)

        # Initialize recurrent carry
        carry = model.init_carry(args.num_envs, rngs)
        prev_action = jnp.zeros(args.num_envs, dtype=jnp.int32)
        prev_reward = jnp.zeros(args.num_envs)

        # === Interleaved collect-update inner loop ===
        trial_reward = jnp.zeros(args.num_envs)
        all_grad_norms = []

        for _ in range(num_inner_updates):
            # Save RNN state before collection (used as init_carry for PPO forward pass)
            init_hstate = jax.tree.map(lambda c: c.copy(), carry)

            # --- Collect num_steps_per_update steps ---
            (timestep, prev_action, prev_reward, carry, rollout_rng), transitions = \
                rollout_scan(model, env, meta_env_params, timestep, prev_action, prev_reward,
                             carry, rollout_rng, args.num_steps_per_update)

            # --- GAE (meta-RL: no episode boundary masking) ---
            next_value = bootstrap_value(model, timestep, prev_action, prev_reward, carry)
            advantages, returns = calculate_gae(transitions, next_value,
                                                gamma=args.gamma, lmbda=args.lmbda)

            # Accumulate rewards for logging
            trial_reward = trial_reward + transitions.reward.sum(axis=0)

            # --- PPO update ---
            for _ in range(args.num_epochs):
                rng, perm_rng = jax.random.split(rng)
                env_indices = jax.random.permutation(perm_rng, args.num_envs)
                env_indices = env_indices[:args.num_minibatches * envs_per_batch]
                minibatches = make_minibatches(transitions, advantages, returns,
                                               init_hstate, env_indices, envs_per_batch)
                grad_norms = update_ppo(model, optimizer, minibatches, metrics,
                                        clip_eps=args.clip_eps, ent_coef=args.ent_coef)
                all_grad_norms.append(grad_norms)

        global_env_step += transitions_per_iter

        # === Logging ===
        iter_elapsed = time.time() - iter_start
        sps = transitions_per_iter / iter_elapsed

        train_reward_mean = float(trial_reward.mean())
        mean_grad_norm = float(jnp.mean(jnp.stack(all_grad_norms)))

        metric_values = {k: float(v) for k, v in metrics.compute().items()}
        wandb_payload = {
            "train/iteration": iteration,
            "train/global_env_step": global_env_step,
            "train/actor_loss": metric_values["actor_loss"],
            "train/critic_loss": metric_values["critic_loss"],
            "train/entropy_loss": metric_values["entropy_loss"],
            "train/reward_mean": train_reward_mean,
            "train/sps": sps,
            "train/grad_norm": mean_grad_norm,
        }

        # === Evaluation on test benchmark (every eval_interval transitions) ===
        if global_env_step - last_eval_step >= args.eval_interval:
            last_eval_step = global_env_step
            rng, eval_run_rng = jax.random.split(rng)

            eval_rng_keys = jax.random.split(eval_run_rng, args.eval_num_envs)

            eval_stats = jax.vmap(
                lambda rng_k, params: evaluate(model, env, params, rng_k, args.episodes_per_trial),
                in_axes=(0, 0),
            )(eval_rng_keys, eval_env_params)

            wandb_payload["test/reward_mean"] = float(eval_stats.reward_sum.mean())
            wandb_payload["test/reward_median"] = float(jnp.median(eval_stats.reward_sum))
            wandb_payload["test/length_mean"] = float(eval_stats.length_sum.mean())

            # === Save checkpoint ===
            _, state = nnx.split(model)
            ckpt_mngr.save(global_env_step, args=ocp.args.StandardSave(state))

        wandb.log(wandb_payload, step=global_env_step)
        metrics.reset()
        iteration += 1

    # === Save final checkpoint ===
    _, state = nnx.split(model)
    ckpt_mngr.save(global_env_step, args=ocp.args.StandardSave(state))
    ckpt_mngr.wait_until_finished()


if __name__ == "__main__":
    main()
