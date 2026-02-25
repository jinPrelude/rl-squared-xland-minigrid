import flax.nnx as nnx
from flax import struct
import jax
from jax import numpy as jnp
import numpy as np
import optax
import rlax


class Transition(struct.PyTreeNode):
    obs_img: jax.Array
    obs_dir: jax.Array
    prev_action: jax.Array
    prev_reward: jax.Array
    done: jax.Array        # actual episode boundary (fed to LSTM)
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
    """Collect a rollout using jax.lax.scan. All env interactions are vmapped across envs.

    Args:
        model: RL2LSTM (NNX module, captured in closure)
        env: wrapped XLand-MiniGrid environment
        env_params: batched env params with vmapped rulesets
        timestep: batched TimeStep from reset
        prev_action: (num_envs,) int32
        prev_reward: (num_envs,) float32
        carry: LSTM carry (h, c) each (num_envs, hidden_dim)
        rng_key: JAX PRNG key
        num_steps: number of scan steps (trial_length)

    Returns:
        final_state: (timestep, prev_action, prev_reward, carry, rng)
        transitions: Transition with shape (num_steps, num_envs, ...)
    """
    def _step_fn(scan_carry, _):
        timestep, prev_action, prev_reward, carry, rng = scan_carry
        rng, action_rng = jax.random.split(rng)

        # Episode boundary signal for RL^2 (model input)
        done = timestep.last().astype(jnp.float32)

        # Forward pass (model captured in closure, read-only)
        logits, value, new_carry = model.step(
            timestep.observation["img"],
            timestep.observation["direction"],
            prev_action, prev_reward, done, carry,
        )

        # Sample action
        log_probs_all = jax.nn.log_softmax(logits, axis=-1)
        action = jax.random.categorical(action_rng, logits, axis=-1)
        log_prob = jnp.take_along_axis(log_probs_all, action[..., None], axis=-1).squeeze(-1)

        # Step environment (vmapped over envs)
        new_timestep = jax.vmap(env.step, in_axes=(0, 0, 0))(env_params, timestep, action)

        transition = Transition(
            obs_img=timestep.observation["img"],
            obs_dir=timestep.observation["direction"],
            prev_action=prev_action,
            prev_reward=prev_reward,
            done=done,
            action=action,
            log_prob=log_prob,
            value=value.squeeze(-1),
            reward=new_timestep.reward,
        )

        new_scan_carry = (new_timestep, action, new_timestep.reward, new_carry, rng)
        return new_scan_carry, transition

    init_carry = (timestep, prev_action, prev_reward, carry, rng_key)
    final_carry, transitions = jax.lax.scan(_step_fn, init_carry, None, length=num_steps)
    return final_carry, transitions


@nnx.jit
def bootstrap_value(model, timestep, prev_action, prev_reward, carry):
    done = timestep.last().astype(jnp.float32)
    _, value, _ = model.step(
        timestep.observation["img"],
        timestep.observation["direction"],
        prev_action, prev_reward, done, carry,
    )
    return value.squeeze(-1)


def calculate_gae(transitions, next_value, gamma, lmbda):
    """Calculate GAE. For meta-RL, done is always 0 (entire trial = one episode)."""
    values = transitions.value
    rewards = transitions.reward
    # Meta-RL convention: no episode boundary masking for GAE
    next_values = jnp.concatenate([values[1:], next_value[None, :]], axis=0)
    deltas = rewards + gamma * next_values - values

    def scan_step(last_advantage, delta_t):
        advantage = delta_t + gamma * lmbda * last_advantage
        return advantage, advantage

    init_advantage = jnp.zeros_like(next_value)
    _, advantages = jax.lax.scan(scan_step, init_advantage, deltas, reverse=True)
    returns = advantages + values
    return advantages, returns


def loss_fn(model, batch, clip_eps, ent_coef):
    (obs_img, obs_dir, prev_action, prev_reward, done,
     actions, old_log_probs, advantages, returns, init_carry) = batch

    logits, values, _ = model.unroll(obs_img, obs_dir, prev_action, prev_reward, done, init_carry)
    values = values.squeeze(-1)

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    selected_log_probs = jnp.take_along_axis(log_probs, actions[..., None], axis=-1).squeeze(-1)

    ratio = jnp.exp(selected_log_probs - old_log_probs)
    actor_loss = rlax.clipped_surrogate_pg_loss(ratio.reshape(-1), advantages.reshape(-1), clip_eps).mean()
    critic_loss = optax.huber_loss(values, jax.lax.stop_gradient(returns)).mean()

    # Entropy bonus (negative because we maximize entropy)
    probs = jax.nn.softmax(logits, axis=-1)
    entropy = -(probs * log_probs).sum(axis=-1).mean()
    entropy_loss = -entropy

    total_loss = actor_loss + 0.5 * critic_loss + ent_coef * entropy_loss
    return total_loss, (actor_loss, critic_loss, entropy_loss)


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


def make_minibatches(transitions, advantages, returns, initial_carry, env_indices, envs_per_batch):
    """Partition transitions along env dimension into minibatches."""
    num_mb = env_indices.shape[0] // envs_per_batch
    idx = env_indices.reshape(num_mb, envs_per_batch)

    def _gather(arr):
        # arr: (T, E, ...) -> (num_mb, T, envs_per_batch, ...)
        return jnp.swapaxes(arr[:, idx], 0, 1)

    return (
        _gather(transitions.obs_img),
        _gather(transitions.obs_dir),
        _gather(transitions.prev_action),
        _gather(transitions.prev_reward),
        _gather(transitions.done),
        _gather(transitions.action),
        _gather(transitions.log_prob),
        _gather(advantages),
        _gather(returns),
        jax.tree.map(lambda c: c[idx], initial_carry),
    )


def evaluate(model, env, env_params, rng_key, num_episodes):
    """Evaluate agent over num_episodes consecutive episodes on a single env.

    Uses jax.lax.while_loop. Model captured in closure (read-only).
    Should be vmapped externally for multiple eval envs.
    """
    def _cond_fn(carry):
        _, stats, _, _, _, _ = carry
        return stats.episodes_done < num_episodes

    def _body_fn(carry):
        rng, stats, timestep, prev_action, prev_reward, hstate = carry
        rng, action_rng = jax.random.split(rng)

        done = timestep.last().astype(jnp.float32)

        # Add batch dim for model (expects batched input)
        logits, _, new_hstate = model.step(
            timestep.observation["img"][None],
            timestep.observation["direction"][None],
            prev_action[None],
            prev_reward[None],
            done[None],
            jax.tree.map(lambda x: x[None], hstate),
        )
        # Remove batch dim
        logits = logits[0]
        new_hstate = jax.tree.map(lambda x: x[0], new_hstate)

        action = jax.random.categorical(action_rng, logits, axis=-1)
        new_timestep = env.step(env_params, timestep, action)

        stats = stats.replace(
            reward_sum=stats.reward_sum + new_timestep.reward,
            length_sum=stats.length_sum + 1,
            episodes_done=stats.episodes_done + new_timestep.last().astype(jnp.int32),
        )
        return (rng, stats, new_timestep, action, new_timestep.reward, new_hstate)

    reset_rng, eval_rng = jax.random.split(rng_key)
    timestep = env.reset(env_params, reset_rng)

    # Single-env LSTM carry (no batch dim)
    init_hstate = jax.tree.map(
        lambda x: x[0],
        model.init_carry(1, nnx.Rngs(0)),
    )

    init_carry = (
        eval_rng,
        RolloutStats(),
        timestep,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        init_hstate,
    )
    final_carry = jax.lax.while_loop(_cond_fn, _body_fn, init_carry)
    return final_carry[1]
