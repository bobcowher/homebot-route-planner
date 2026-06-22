"""Greedy evaluation of downloaded checkpoints.

Runs N fully-greedy episodes (epsilon=0) per checkpoint and reports success
percentage. Env config matches train.py exactly so the numbers are comparable
to the training chart minus the epsilon-0.1 exploration noise.

Usage:
    python3 evaluate.py                  # 100 episodes, q_model.pt + best.pt
    python3 evaluate.py --episodes 20
    python3 evaluate.py --checkpoints checkpoints/q_model.pt
"""

import argparse
import random

import cv2
import gymnasium as gym
import torch

import homebot  # noqa: F401  (side-effect env registration)
from goal_geometry import world_coords
from motion import MotionState
from models.q_model import QModel


def make_env():
    env = gym.make(
        "HomeBot2D-Goal-V1",
        render_mode="rgb_array",
        action_mode="discrete",
        obs_resolution=(96, 96),
        n_trash=2,
        max_steps=1000,
        map_name="default",
        goals=["collect_trash"],
        random_start=True,   # match train.py spawn
    )
    return env


def load_q_model(path, n_actions, device, goal_layers=1, head_layers=1, head_norm=False,
                 use_motion=False, motion_window=1, macro_h=1):
    """n_actions is the BASE action count. For a macro checkpoint (macro_h>1) the
    output head is n_actions ** macro_h; macro_h is read from the checkpoint meta when
    present so callers need not know it. The returned model carries macro_h / n_base
    so every eval path can decode an output index back to base actions."""
    state = torch.load(path, map_location=device)
    if "q_model" in state:  # best.pt wraps the state_dict with metadata
        print(f"  ({path} is a best-checkpoint from episode {state.get('episode')})")
        macro_h = state.get("macro_h", macro_h)
        state = state["q_model"]
    model = QModel(action_dim=n_actions ** macro_h, goal_layers=goal_layers,
                   head_layers=head_layers, head_norm=head_norm,
                   use_motion=use_motion, motion_window=motion_window,
                   macro_h=macro_h, n_base=n_actions).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def process_observation(obs):
    obs = cv2.resize(obs, (96, 96), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(obs).permute(2, 0, 1)


def evaluate(model, env, episodes, device, epsilon=0.0):
    successes = 0
    success_steps = []

    use_motion = getattr(model, "use_motion", False)
    motion_window = getattr(model, "motion_window", 1)
    for episode in range(episodes):
        raw_obs, _ = env.reset()
        obs = process_observation(raw_obs["observation"])
        desired_goal = raw_obs["desired_goal"]
        pos = raw_obs["achieved_goal"]
        ms = MotionState(env.action_space.n, motion_window)  # type: ignore[attr-defined]

        done = False
        steps = 0
        episode_reward = 0.0

        while not done:
            if epsilon > 0 and random.random() < epsilon:
                action = env.action_space.sample()
                ms.commit(pos[0], pos[1], int(action))
            else:
                with torch.no_grad():
                    obs_t = obs.unsqueeze(0).float().to(device) / 255.0
                    # Coord rep: [robot_x, robot_y, goal_x, goal_y], pose updates each step.
                    goal_vec = world_coords(pos[0], pos[1],
                                            desired_goal[0], desired_goal[1])
                    goal_t = torch.as_tensor(
                        goal_vec, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    motion_t = None
                    if use_motion:
                        motion_t = torch.as_tensor(ms.vec(pos[0], pos[1]),
                                                   dtype=torch.float32, device=device).unsqueeze(0)
                    action = model(obs_t, goal_t, motion_t).argmax(dim=1).item()
                    ms.commit(pos[0], pos[1], int(action))
            raw_next, reward, term, trunc, _ = env.step(action)
            obs = process_observation(raw_next["observation"])
            pos = raw_next["achieved_goal"]
            done = term or trunc
            episode_reward += float(reward)
            steps += 1

        if episode_reward > 0.5:
            successes += 1
            success_steps.append(steps)
        print(f"Episode {episode} | reward: {episode_reward:.1f} | steps: {steps}")

    pct = 100.0 * successes / episodes
    avg_steps = sum(success_steps) / len(success_steps) if success_steps else float("nan")
    print(f"\nSuccess: {successes}/{episodes} = {pct:.0f}% | avg steps on success: {avg_steps:.0f}")
    return pct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--checkpoints", nargs="+",
        default=["checkpoints/q_model.pt"],
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.0,
        help="random-action rate; 0.1 reproduces training conditions",
    )
    parser.add_argument("--goal-layers", type=int, default=1)
    parser.add_argument("--head-layers", type=int, default=1)
    parser.add_argument("--head-norm", action="store_true")
    parser.add_argument("--use-motion", action="store_true")
    parser.add_argument("--motion-window", type=int, default=1,
                        help="windowed net-displacement horizon the checkpoint was "
                             "trained with (1 = original velocity-only motion)")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = make_env()
    n_actions = env.action_space.n  # type: ignore[union-attr]

    results = {}
    for path in args.checkpoints:
        print(f"\n=== {path} | {args.episodes} episodes | epsilon={args.epsilon} ===")
        model = load_q_model(path, n_actions, device,
                             goal_layers=args.goal_layers, head_layers=args.head_layers,
                             head_norm=args.head_norm, use_motion=args.use_motion,
                             motion_window=args.motion_window)
        results[path] = evaluate(model, env, args.episodes, device, args.epsilon)

    print("\n=== Summary ===")
    for path, pct in results.items():
        print(f"{path}: {pct:.0f}%")


if __name__ == "__main__":
    main()
