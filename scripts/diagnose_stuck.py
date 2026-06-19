"""Classify the navigator's failure MODE on timed-out chain legs (headless).

reach/timeout (chained_eval) tells us WHERE a chain breaks; this tells us HOW.
For every leg, log the per-step (action, x, y, dist) trace. On a timed-out leg,
classify the trace into one of:

  WALL_STICK   - mean per-step displacement ~0: commanded moves aren't
                 translating to motion (robot pushing into a wall/fixture, argmax
                 re-picks the into-wall action every step). This is the pin the
                 blocked-move penalty targets.
  OSCILLATION  - robot keeps moving each step but stays confined to a tiny
                 region: an A-B-A-B limit cycle (bang-bang vibration).
  WANDER       - moving and roaming a wide area but never closing on the goal:
                 a navigation/value failure, not a motor lock.

Thresholds are in ROBOT_STEP_PX units so they track the env's step size.

    python3 scripts/diagnose_stuck.py --checkpoint checkpoints/run306_q_model_best.pt \
        --goal-layers 2 --head-layers 4 --use-motion --episodes 10
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import gymnasium as gym
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import homebot  # noqa: F401  (env registration)
from homebot.goals import GOAL_THRESHOLD
from evaluate import load_q_model, process_observation
from goal_geometry import distance, eval_step_budget, ROBOT_STEP_PX
from motion import MotionState
from chained_eval import _select_action, REACH_OVERRIDE
from task_chain import DEFAULT_CHAIN, resolve_goal


def classify(trace):
    """trace: list of (action, x, y, dist). Returns (label, stats dict)."""
    xs = [p[1] for p in trace]
    ys = [p[2] for p in trace]
    steps = [distance(xs[i], ys[i], xs[i + 1], ys[i + 1]) for i in range(len(trace) - 1)]
    mean_step = sum(steps) / len(steps) if steps else 0.0

    # Spatial extent of the back half (where a stuck policy settles into its mode).
    half = trace[len(trace) // 2:]
    hx, hy = [p[1] for p in half], [p[2] for p in half]
    span = max((distance(a, b, c, d) for a in hx for b in hy
                for c in hx for d in hy), default=0.0) if len(half) > 1 else 0.0

    net = trace[0][3] - trace[-1][3]  # dist closed over the leg (+ = got closer)
    stats = {"mean_step_px": mean_step, "span_px": span, "net_closed_px": net,
             "steps": len(trace)}

    if mean_step < 0.5 * ROBOT_STEP_PX:
        return "WALL_STICK", stats
    if span < 4.0 * ROBOT_STEP_PX:
        return "OSCILLATION", stats
    return "WANDER", stats


def run_leg_logged(model, env, base, obs, goal_xy, budget, device, ms, reach):
    robot = base._robot
    trace = []
    action = 0
    for _ in range(1, budget + 1):
        motion = ms.vec(robot.x, robot.y)
        action = _select_action(model, obs, goal_xy, robot, device, "greedy", 0.01, motion)
        ms.commit(robot.x, robot.y, action)
        d = distance(robot.x, robot.y, goal_xy[0], goal_xy[1])
        trace.append((action, robot.x, robot.y, d))
        obs = process_observation(env.step(action)[0])
        if distance(robot.x, robot.y, goal_xy[0], goal_xy[1]) <= reach:
            return True, trace, obs
    # final pose row so net_closed reflects the whole leg
    trace.append((action, robot.x, robot.y,
                  distance(robot.x, robot.y, goal_xy[0], goal_xy[1])))
    return False, trace, obs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/run306_q_model_best.pt")
    p.add_argument("--goal-layers", type=int, default=2)
    p.add_argument("--head-layers", type=int, default=4)
    p.add_argument("--head-norm", action="store_true")
    p.add_argument("--use-motion", action="store_true")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = gym.make("HomeBot2D-V1", render_mode="rgb_array", action_mode="discrete",
                   obs_resolution=(96, 96), n_trash=2, max_steps=20000,
                   map_name="default", random_start=True)
    base = env.unwrapped
    model = load_q_model(args.checkpoint, env.action_space.n, device,
                         goal_layers=args.goal_layers, head_layers=args.head_layers,
                         head_norm=args.head_norm, use_motion=args.use_motion)

    modes = Counter()
    failed = []  # (ep, leg_name, label, stats)
    for ep in range(args.episodes):
        obs = process_observation(env.reset(seed=args.seed + ep)[0])
        r = base._robot
        ms = MotionState(env.action_space.n)
        targets = [(name, resolve_goal(base, name)) for name in DEFAULT_CHAIN]
        for name, (gx, gy) in targets:
            budget = eval_step_budget(distance(r.x, r.y, gx, gy))
            reach = REACH_OVERRIDE.get(name, GOAL_THRESHOLD)
            reached, trace, obs = run_leg_logged(model, env, base, obs, (gx, gy),
                                                 budget, device, ms, reach)
            if not reached:
                label, stats = classify(trace)
                modes[label] += 1
                failed.append((ep, name, label, stats))

    print(f"\n=== failure-mode diagnosis | {args.episodes} episodes | "
          f"chain={DEFAULT_CHAIN} ===")
    if not failed:
        print("no timed-out legs — nothing stuck to diagnose.")
        return
    for ep, name, label, s in failed:
        print(f"  ep{ep} {name:<14} {label:<11} "
              f"mean_step={s['mean_step_px']:.2f}px span={s['span_px']:.0f}px "
              f"net_closed={s['net_closed_px']:+.0f}px steps={s['steps']}")
    print("\n  mode tally:")
    for label, n in modes.most_common():
        print(f"    {label:<11} {n}")


if __name__ == "__main__":
    main()
