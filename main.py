import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import flax.nnx as nnx
import jax
from jax import numpy as jnp
import optax
import orbax.checkpoint as ocp
import wandb
import xminigrid

from shared import make_env, make_model
from rl_squared import rollout_scan, bootstrap_value, calculate_gae, make_minibatches, update_ppo, evaluate


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--env-id", type=str, default="XLand-MiniGrid-R1-9x9")
    parser.add_argument("--benchmark-id", type=str, default="small-1m")
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-transitions", type=int, default=10_000_000_000)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--num-steps-per-env", type=int, default=2560)
    parser.add_argument("--num-steps-per-update", type=int, default=256)
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
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "gru"])
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eval-interval", type=int, default=100_000_000)
    parser.add_argument("--eval-num-envs", type=int, default=4096)
    parser.add_argument("--save-ckpt-dir", type=str, default="./checkpoints")
    return parser.parse_args()


def main():
    args = parse_arguments()
    assert args.num_envs % args.num_minibatches == 0
    envs_per_batch = args.num_envs // args.num_minibatches

    env, env_params = make_env(args.env_id)
    num_actions = env.num_actions(env_params)

    assert args.num_steps_per_env % args.num_steps_per_update == 0
    num_inner_updates = args.num_steps_per_env // args.num_steps_per_update

    benchmark = xminigrid.load_benchmark(args.benchmark_id)
    rng = jax.random.key(args.seed)
    rng, shuffle_rng = jax.random.split(rng)
    benchmark = benchmark.shuffle(shuffle_rng)
    train_benchmark, test_benchmark = benchmark.split(args.train_split)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"rl2_{args.model}_{args.env_id}_{timestamp}"
    wandb.init(project="rl-squared-xland", name=run_name, config=vars(args))

    rngs = nnx.Rngs(args.seed)
    model = make_model(args, num_actions, rngs)
    optimizer = nnx.Optimizer(model, optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adam(args.learning_rate),
    ), wrt=nnx.Param)

    metrics = nnx.metrics.MultiMetric(
        critic_loss=nnx.metrics.Average("critic_loss"),
        actor_loss=nnx.metrics.Average("actor_loss"),
        entropy_loss=nnx.metrics.Average("entropy_loss"),
    )

    ckpt_dir = (Path(args.save_ckpt_dir) / run_name).resolve()
    ckpt_mngr = ocp.CheckpointManager(ckpt_dir)

    def save_ckpt():
        _, state = nnx.split(model)
        ckpt_mngr.save(global_env_step, args=ocp.args.StandardSave(state))

    rng, eval_ruleset_rng = jax.random.split(rng)
    eval_ruleset_keys = jax.random.split(eval_ruleset_rng, args.eval_num_envs)
    eval_rulesets = jax.vmap(test_benchmark.sample_ruleset)(eval_ruleset_keys)
    eval_env_params = env_params.replace(ruleset=eval_rulesets)

    global_env_step = 0
    last_eval_step = 0
    transitions_per_iter = args.num_steps_per_env * args.num_envs

    while global_env_step < args.total_transitions:
        iter_start = time.time()
        rng, ruleset_rng, reset_rng, rollout_rng = jax.random.split(rng, 4)

        ruleset_keys = jax.random.split(ruleset_rng, args.num_envs)
        rulesets = jax.vmap(train_benchmark.sample_ruleset)(ruleset_keys)
        meta_env_params = env_params.replace(ruleset=rulesets)

        reset_keys = jax.random.split(reset_rng, args.num_envs)
        timestep = jax.vmap(env.reset, in_axes=(0, 0))(meta_env_params, reset_keys)

        carry = model.init_carry(args.num_envs, rngs)
        prev_action = jnp.zeros(args.num_envs, dtype=jnp.int32)
        prev_reward = jnp.zeros(args.num_envs)

        trial_reward = jnp.zeros(args.num_envs)
        all_grad_norms = []

        for _ in range(num_inner_updates):
            init_hstate = carry

            (timestep, prev_action, prev_reward, carry, rollout_rng), transitions = \
                rollout_scan(model, env, meta_env_params, timestep, prev_action, prev_reward,
                             carry, rollout_rng, args.num_steps_per_update)

            next_value = bootstrap_value(model, timestep, prev_action, prev_reward, carry)
            advantages, returns = calculate_gae(transitions, next_value, gamma=args.gamma, lmbda=args.lmbda)
            trial_reward = trial_reward + transitions.reward.sum(axis=0)

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
        metric_values = {k: float(v) for k, v in metrics.compute().items()}
        wandb_payload = {
            "train/global_env_step": global_env_step,
            "train/reward_mean": float(trial_reward.mean()),
            "train/sps": transitions_per_iter / (time.time() - iter_start),
            "train/grad_norm": float(jnp.mean(jnp.stack(all_grad_norms))),
            **{f"train/{k}": v for k, v in metric_values.items()},
        }

        if global_env_step - last_eval_step >= args.eval_interval:
            last_eval_step = global_env_step
            rng, eval_run_rng = jax.random.split(rng)
            eval_rng_keys = jax.random.split(eval_run_rng, args.eval_num_envs)

            eval_stats, _ = jax.vmap(
                lambda rng_k, params: evaluate(model, env, params, rng_k, args.num_steps_per_env),
                in_axes=(0, 0),
            )(eval_rng_keys, eval_env_params)

            wandb_payload["test/reward_mean"] = float(eval_stats.reward_sum.mean())
            wandb_payload["test/reward_median"] = float(jnp.median(eval_stats.reward_sum))
            wandb_payload["test/length_mean"] = float(eval_stats.length_sum.mean())
            save_ckpt()

        wandb.log(wandb_payload, step=global_env_step)
        metrics.reset()

    save_ckpt()
    ckpt_mngr.wait_until_finished()


if __name__ == "__main__":
    main()
