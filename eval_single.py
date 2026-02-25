"""
eval_single.py - Meta-RL single-environment evaluation with video recording.

Loads a trained RL² checkpoint, runs a full trial (multiple episodes on one
ruleset from the test benchmark), and produces an MP4 video showing the agent's
behavior with a HUD overlay and an episodic reward plot.

Usage:
    python eval_single.py --ckpt-dir checkpoints/rl2_gru_XLand-MiniGrid-R1-9x9_20260225_160340
"""

from argparse import ArgumentParser
from pathlib import Path

import flax.nnx as nnx
import imageio.v2 as imageio
import jax
from jax import numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint as ocp
from PIL import Image, ImageDraw
from tqdm import tqdm
import xminigrid
from xminigrid.wrappers import GymAutoResetWrapper, DirectionObservationWrapper
from xminigrid.rendering.text_render import _text_encode_goal, print_ruleset

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
    parser.add_argument("--env-index", type=int, default=None,
                        help="env_index from eval_all.py CSV. Reproduces the exact same ruleset.")
    parser.add_argument("--eval-seed", type=int, default=42,
                        help="Must match eval_all.py --eval-seed (default 42)")
    parser.add_argument("--eval-num-envs", type=int, default=4096,
                        help="Must match eval_all.py --eval-num-envs (default 4096)")
    parser.add_argument("--rnn-hidden-dim", type=int, default=1024)
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--obs-emb-dim", type=int, default=16)
    parser.add_argument("--action-emb-dim", type=int, default=16)
    parser.add_argument("--episodes-per-trial", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true",
                        help="Use argmax instead of sampling.")
    parser.add_argument("--output", type=str, default="eval_video.mp4")
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Frame composition
# ---------------------------------------------------------------------------

_cached_episode_plot = {"n": -1, "img": None}


def render_episode_plot(completed_rewards, total_episodes, width, height=100):
    """Render episodic reward bar chart. X-axis expands dynamically."""
    n = len(completed_rewards)
    if n == _cached_episode_plot["n"] and _cached_episode_plot["img"] is not None:
        return _cached_episode_plot["img"]

    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    x_max = max(total_episodes, n)
    ax.set_xlim(0.5, x_max + 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Episode", fontsize=7)
    ax.set_ylabel("Return", fontsize=7)
    ax.tick_params(labelsize=6)

    if x_max <= 20:
        ax.set_xticks(range(1, x_max + 1))
    else:
        tick_step = max(1, x_max // 10)
        ticks = list(range(1, x_max + 1, tick_step))
        if ticks[-1] != x_max:
            ticks.append(x_max)
        ax.set_xticks(ticks)

    if completed_rewards:
        episodes = list(range(1, n + 1))
        ax.bar(episodes, completed_rewards, color="steelblue", width=0.7)

    fig.tight_layout()
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    plot_img = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)

    if plot_img.shape[1] != width:
        pil_img = Image.fromarray(plot_img)
        pil_img = pil_img.resize((width, height), Image.LANCZOS)
        plot_img = np.array(pil_img)

    _cached_episode_plot["n"] = n
    _cached_episode_plot["img"] = plot_img
    return plot_img


def compose_frame(env_frame, episode_num, step_in_episode,
                  current_reward, completed_rewards, total_episodes,
                  goal_text=""):
    W = env_frame.shape[1]

    # HUD text bar
    hud_height = 54
    hud = np.full((hud_height, W, 3), 40, dtype=np.uint8)
    hud_img = Image.fromarray(hud)
    draw = ImageDraw.Draw(hud_img)
    text = f"Ep: {episode_num + 1}/{total_episodes}  Step: {step_in_episode}  R: {current_reward:.2f}"
    draw.text((5, 5), text, fill=(255, 255, 255))
    if goal_text:
        draw.text((5, 25), f"Goal: {goal_text}", fill=(255, 255, 100))
    hud = np.array(hud_img)

    # Episode reward plot
    episode_img = render_episode_plot(completed_rewards, total_episodes, width=W)

    return np.concatenate([env_frame, hud, episode_img], axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_arguments()

    # --- Environment ---
    env, env_params = xminigrid.make(args.env_id)
    env = GymAutoResetWrapper(env)
    env = DirectionObservationWrapper(env)
    num_actions = env.num_actions(env_params)
    max_episode_steps = env_params.max_steps

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

    # --- Ruleset from test benchmark ---
    benchmark = xminigrid.load_benchmark(args.benchmark_id)
    # Match training shuffle (default seed=0)
    benchmark = benchmark.shuffle(jax.random.key(0))
    _, test_benchmark = benchmark.split(args.train_split)

    rng = jax.random.key(args.seed)

    if args.env_index is not None:
        # Reproduce the exact ruleset from eval_all.py for the given env_index
        eval_rng = jax.random.key(args.eval_seed)
        _, ruleset_rng, _ = jax.random.split(eval_rng, 3)
        eval_ruleset_keys = jax.random.split(ruleset_rng, args.eval_num_envs)
        ruleset = test_benchmark.sample_ruleset(eval_ruleset_keys[args.env_index])
        print(f"Using env_index={args.env_index} from eval_all.py (eval_seed={args.eval_seed})")
    else:
        rng, sample_rng = jax.random.split(rng)
        ruleset = test_benchmark.sample_ruleset(sample_rng)

    eval_env_params = env_params.replace(ruleset=ruleset)

    # --- Decode goal for HUD display ---
    goal_text = _text_encode_goal(ruleset.goal.tolist())
    print(f"Goal: {goal_text}")
    print_ruleset(ruleset)

    # --- JIT-compiled step: policy + env.step combined ---
    @jax.jit
    def jit_step(timestep, prev_action, prev_reward, carry, rng):
        rng, action_rng = jax.random.split(rng)
        done = timestep.last().astype(jnp.float32)
        logits, value, new_carry = model.step(
            timestep.observation["img"][None],
            timestep.observation["direction"][None],
            prev_action[None], prev_reward[None], done[None],
            jax.tree.map(lambda x: x[None], carry),
        )
        logits = logits[0]
        new_carry = jax.tree.map(lambda x: x[0], new_carry)

        if args.deterministic:
            action = jnp.argmax(logits, axis=-1)
        else:
            action = jax.random.categorical(action_rng, logits, axis=-1)

        new_timestep = env.step(eval_env_params, timestep, action)
        return new_timestep, action, new_carry, rng

    # --- Reset ---
    rng, reset_rng = jax.random.split(rng)
    timestep = env.reset(eval_env_params, reset_rng)
    carry = jax.tree.map(lambda x: x[0], model.init_carry(1, nnx.Rngs(0)))
    prev_action = jnp.asarray(0, dtype=jnp.int32)
    prev_reward = jnp.asarray(0.0)

    # --- Evaluation loop ---
    total_steps = args.episodes_per_trial * max_episode_steps
    episode_num = 0
    step_in_episode = 0
    episode_reward = 0.0
    episode_rewards = []

    frames = []

    # Warm up JIT (first call triggers compilation)
    print("JIT compiling...")
    new_timestep, action, new_carry, rng = jit_step(timestep, prev_action, prev_reward, carry, rng)

    print(f"Running {args.episodes_per_trial} episodes ({total_steps} steps)...")

    for global_step in tqdm(range(total_steps), desc="Eval"):
        # Render current state
        env_frame = env.render(eval_env_params, timestep)
        frame = compose_frame(env_frame, episode_num, step_in_episode,
                              episode_reward, episode_rewards, args.episodes_per_trial,
                              goal_text=goal_text)
        frames.append(frame)

        # Combined policy + env step (JIT-compiled)
        new_timestep, action, new_carry, rng = jit_step(
            timestep, prev_action, prev_reward, carry, rng,
        )

        step_in_episode += 1
        episode_reward += float(new_timestep.reward)
        # Episode boundary (GymAutoResetWrapper preserves step_type=LAST)
        if bool(new_timestep.last()):
            episode_rewards.append(episode_reward)
            print(f"  Episode {episode_num + 1}: return={episode_reward:.3f}, length={step_in_episode}")
            episode_num += 1
            step_in_episode = 0
            episode_reward = 0.0

        prev_action = action
        prev_reward = new_timestep.reward
        carry = new_carry
        timestep = new_timestep

    # Write video (after JAX loop to avoid os.fork() conflict with JAX threads)
    print(f"Writing {len(frames)} frames to video...")
    imageio.mimwrite(args.output, frames, fps=args.fps, quality=8,
                     macro_block_size=1)

    print(f"Saved video to {args.output}")
    print(f"Episode returns: {[f'{r:.3f}' for r in episode_rewards]}")
    print(f"Mean return: {np.mean(episode_rewards):.3f}")


if __name__ == "__main__":
    main()
