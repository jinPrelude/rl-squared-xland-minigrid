import flax.nnx as nnx
from flax import struct
import jax
from jax import numpy as jnp
import optax
import rlax


class Transition(struct.PyTreeNode):
    obs_img: jax.Array
    obs_dir: jax.Array
    prev_action: jax.Array
    prev_reward: jax.Array
    done: jax.Array
    action: jax.Array
    log_prob: jax.Array
    value: jax.Array
    reward: jax.Array


class RolloutStats(struct.PyTreeNode):
    reward_sum: jax.Array = struct.field(default_factory=lambda: jnp.asarray(0.0))
    length_sum: jax.Array = struct.field(default_factory=lambda: jnp.asarray(0))
    episodes_done: jax.Array = struct.field(default_factory=lambda: jnp.asarray(0))


@nnx.jit(static_argnames=('env', 'num_steps'))
def rollout_scan(model, env, env_params, timestep, prev_action, prev_reward, carry, rng_key, num_steps):
    """Collect num_steps of rollout using jax.lax.scan, vmapped across envs."""
    def _step_fn(scan_carry, _):
        timestep, prev_action, prev_reward, carry, rng = scan_carry
        rng, action_rng = jax.random.split(rng)

        done = timestep.last().astype(jnp.float32)
        logits, value, new_carry = model.step(
            timestep.observation["img"], timestep.observation["direction"],
            prev_action, prev_reward, done, carry,
        )

        log_probs_all = jax.nn.log_softmax(logits, axis=-1)
        action = jax.random.categorical(action_rng, logits, axis=-1)
        log_prob = jnp.take_along_axis(log_probs_all, action[..., None], axis=-1).squeeze(-1)

        new_timestep = jax.vmap(env.step, in_axes=(0, 0, 0))(env_params, timestep, action)

        transition = Transition(
            obs_img=timestep.observation["img"],
            obs_dir=timestep.observation["direction"],
            prev_action=prev_action, prev_reward=prev_reward, done=done,
            action=action, log_prob=log_prob, value=value.squeeze(-1),
            reward=new_timestep.reward,
        )
        return (new_timestep, action, new_timestep.reward, new_carry, rng), transition

    final_carry, transitions = jax.lax.scan(_step_fn, (timestep, prev_action, prev_reward, carry, rng_key), None, length=num_steps)
    return final_carry, transitions


@nnx.jit
def bootstrap_value(model, timestep, prev_action, prev_reward, carry):
    done = timestep.last().astype(jnp.float32)
    _, value, _ = model.step(
        timestep.observation["img"], timestep.observation["direction"],
        prev_action, prev_reward, done, carry,
    )
    return value.squeeze(-1)


def calculate_gae(transitions, next_value, gamma, lmbda):
    """Calculate GAE. Meta-RL: no episode boundary masking (entire trial = one episode)."""
    next_values = jnp.concatenate([transitions.value[1:], next_value[None, :]], axis=0)
    deltas = transitions.reward + gamma * next_values - transitions.value

    def scan_step(last_adv, delta_t):
        adv = delta_t + gamma * lmbda * last_adv
        return adv, adv

    _, advantages = jax.lax.scan(scan_step, jnp.zeros_like(next_value), deltas, reverse=True)
    return advantages, advantages + transitions.value


def loss_fn(model, batch, clip_eps, ent_coef):
    (transitions, advantages, returns), init_carry = batch

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    logits, values, _ = model.unroll(
        transitions.obs_img, transitions.obs_dir, transitions.prev_action,
        transitions.prev_reward, transitions.done, init_carry,
    )
    values = values.squeeze(-1)

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    selected_log_probs = jnp.take_along_axis(log_probs, transitions.action[..., None], axis=-1).squeeze(-1)
    ratio = jnp.exp(selected_log_probs - transitions.log_prob)
    actor_loss = rlax.clipped_surrogate_pg_loss(ratio.reshape(-1), advantages.reshape(-1), clip_eps).mean()

    value_pred_clipped = transitions.value + (values - transitions.value).clip(-clip_eps, clip_eps)
    critic_loss = 0.5 * jnp.maximum(
        jnp.square(values - returns), jnp.square(value_pred_clipped - returns),
    ).mean()

    probs = jax.nn.softmax(logits, axis=-1)
    entropy = -(probs * log_probs).sum(axis=-1).mean()

    total_loss = actor_loss + 0.5 * critic_loss - ent_coef * entropy
    return total_loss, (actor_loss, critic_loss, -entropy)


@nnx.jit
def update_ppo(model, optimizer, minibatches, metrics, clip_eps=0.2, ent_coef=0.01):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_step(carry, minibatch):
        model, optimizer, metrics = carry
        (loss, (actor_loss, critic_loss, entropy_loss)), grad = grad_fn(model, minibatch, clip_eps, ent_coef)
        grad_norm = optax.global_norm(grad)
        optimizer.update(model, grad)
        metrics.update(actor_loss=actor_loss, critic_loss=critic_loss, entropy_loss=entropy_loss)
        return (model, optimizer, metrics), grad_norm

    _, grad_norms = scan_step((model, optimizer, metrics), minibatches)
    return grad_norms


def make_minibatches(transitions, advantages, returns, init_carry, env_indices, envs_per_batch):
    """Partition transitions along env dimension into minibatches."""
    num_mb = env_indices.shape[0] // envs_per_batch
    idx = env_indices.reshape(num_mb, envs_per_batch)
    gather_time = lambda arr: jnp.swapaxes(arr[:, idx], 0, 1)  # (T,E,...) -> (num_mb,T,epb,...)
    return (
        jax.tree.map(gather_time, (transitions, advantages, returns)),
        jax.tree.map(lambda c: c[idx], init_carry),
    )


def evaluate(model, env, env_params, rng_key, num_steps):
    """Evaluate agent over num_steps steps. Vmap externally for batch eval."""
    def _step_fn(carry, _):
        rng, timestep, prev_action, prev_reward, hstate, stats = carry
        rng, action_rng = jax.random.split(rng)

        done = timestep.last().astype(jnp.float32)
        logits, _, new_hstate = model.step(
            timestep.observation["img"][None], timestep.observation["direction"][None],
            prev_action[None], prev_reward[None], done[None],
            jax.tree.map(lambda x: x[None], hstate),
        )
        logits = logits[0]
        new_hstate = jax.tree.map(lambda x: x[0], new_hstate)

        action = jax.random.categorical(action_rng, logits, axis=-1)
        new_timestep = env.step(env_params, timestep, action)

        stats = stats.replace(
            reward_sum=stats.reward_sum + new_timestep.reward,
            length_sum=stats.length_sum + 1,
            episodes_done=stats.episodes_done + new_timestep.last().astype(jnp.int32),
        )
        return (rng, new_timestep, action, new_timestep.reward, new_hstate, stats), new_timestep.reward

    reset_rng, eval_rng = jax.random.split(rng_key)
    timestep = env.reset(env_params, reset_rng)
    init_hstate = jax.tree.map(lambda x: x[0], model.init_carry(1, nnx.Rngs(0)))

    init_carry = (
        eval_rng, timestep,
        jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0.0), init_hstate, RolloutStats(),
    )
    (_, _, _, _, _, stats), step_rewards = jax.lax.scan(_step_fn, init_carry, None, length=num_steps)
    return stats, step_rewards
