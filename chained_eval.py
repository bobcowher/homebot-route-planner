"""Chained multi-goal evaluation in the non-goal env.

The deploy story (SayCan): an upstream planner decomposes a task into an ordered
list of go-to(coord) subgoals; the learned navigator executes them. This harness
is the eval half — it walks a STATIC orchestrated list of named goals through the
non-goal env (HomeBot2D-V1) in ONE episode, with the robot pose persisting from
one leg to the next. The static list stands in for the LLM, so navigator quality
is measured independent of any planner.

Reach criterion is GOAL_THRESHOLD (79px) — the same threshold the env reward uses
in the goal env, so chain legs are scored exactly as the navigator was trained.

Usage:
    python3 chained_eval.py                              # all-5 sweep, both readouts
    python3 chained_eval.py --episodes 20 --checkpoint <depth4>.pt --goal-layers 2 --head-layers 4
    python3 chained_eval.py --chain collect_trash deliver_package deliver_drink
    python3 chained_eval.py --chain go_to_fridge         # single-leg sanity check
"""

import argparse

import gymnasium as gym
import torch
import torch.nn.functional as F

import homebot  # noqa: F401  (side-effect env registration)
from homebot.goals import GOAL_NAMES, GOAL_THRESHOLD
from evaluate import load_q_model, process_observation
from goal_geometry import world_coords, distance, eval_step_budget

# All-5 sweep: every named location once. collect_trash last so any mid-leg trash
# removal can't disturb earlier legs' coordinate resolution.
DEFAULT_CHAIN = ["go_to_fridge", "deliver_drink", "go_to_door",
                 "deliver_package", "collect_trash"]


def _select_action(model, obs, goal_xy, robot, device, readout, temp):
    """One greedy/softmax action from the current obs + pose toward goal_xy."""
    with torch.no_grad():
        obs_t = obs.unsqueeze(0).float().to(device) / 255.0
        goal_vec = world_coords(robot.x, robot.y, goal_xy[0], goal_xy[1])
        goal_t = torch.as_tensor(goal_vec, dtype=torch.float32,
                                 device=device).unsqueeze(0)
        q = model(obs_t, goal_t).squeeze(0)
        if readout == "softmax":
            probs = F.softmax(q / temp, dim=0)
            return int(torch.multinomial(probs, 1).item())
        return int(q.argmax().item())


def run_leg(model, env, base, obs, goal_xy, budget, device, readout, temp):
    """Drive the navigator toward goal_xy until reached or budget exhausted.

    Returns (reached, steps, obs). obs is threaded back out so the next leg
    continues from the live observation without an env reset.
    """
    robot = base._robot
    for steps in range(1, budget + 1):
        action = _select_action(model, obs, goal_xy, robot, device, readout, temp)
        next_obs, _, _, _, _ = env.step(action)
        obs = process_observation(next_obs)
        if distance(robot.x, robot.y, goal_xy[0], goal_xy[1]) <= GOAL_THRESHOLD:
            return True, steps, obs
    return False, budget, obs


def run_chain(model, env, chain, device, readout, temp, seed, budget_mult=1.0):
    """Reset once, walk the chain leg-by-leg. Pose persists across legs; a failed
    leg does NOT abort the chain (we continue so we can see where it breaks).

    budget_mult scales the per-leg step budget (1.0 = goal_geometry's default
    anti-circling budget; larger relaxes it toward the training eval's full
    episode length for comparability).

    Returns list of (name, reached, steps) per leg.
    """
    base = env.unwrapped
    raw_obs, _ = env.reset(seed=seed)
    obs = process_observation(raw_obs)
    robot = base._robot

    # Resolve every leg's target coordinate up front (the static orchestrated list
    # is coords fixed at plan time). Must happen before stepping: the robot picks
    # up trash incidentally while traversing earlier legs, which would empty
    # trash_positions and make a later collect_trash leg unresolvable.
    targets = [(name, base.goal_to_coordinates(name)) for name in chain]

    results = []
    for name, (gx, gy) in targets:
        budget = max(1, int(eval_step_budget(distance(robot.x, robot.y, gx, gy)) * budget_mult))
        reached, steps, obs = run_leg(model, env, base, obs, (gx, gy),
                                      budget, device, readout, temp)
        results.append((name, reached, steps))
    return results


def _print_readout(label, episodes_results, chain):
    """episodes_results: list (per episode) of list of (name, reached, steps)."""
    n_ep = len(episodes_results)
    print(f"\n=== readout: {label} | {n_ep} episode(s) ===")

    # Per-leg reach counts across episodes.
    per_leg_reached = {name: 0 for name in chain}
    chain_completions = []   # legs reached / total, per episode
    full_chain = 0           # episodes with every leg reached

    for ep_i, legs in enumerate(episodes_results):
        reached_here = sum(1 for _, r, _ in legs if r)
        chain_completions.append(reached_here / len(legs))
        if reached_here == len(legs):
            full_chain += 1
        for name, r, _ in legs:
            if r:
                per_leg_reached[name] += 1
        if n_ep <= 3:  # detailed leg table only for small runs
            print(f"  episode {ep_i}:")
            for name, r, steps in legs:
                mark = "reached" if r else "TIMEOUT"
                print(f"    {name:<16} {mark:<8} steps={steps}")

    print("  per-leg reach rate:")
    for name in chain:
        print(f"    {name:<16} {per_leg_reached[name]}/{n_ep} "
              f"= {100.0 * per_leg_reached[name] / n_ep:.0f}%")
    mean_completion = 100.0 * sum(chain_completions) / n_ep
    print(f"  chain completion (mean legs reached): {mean_completion:.0f}%")
    print(f"  full-chain success (all legs): {full_chain}/{n_ep} "
          f"= {100.0 * full_chain / n_ep:.0f}%")
    return mean_completion, 100.0 * full_chain / n_ep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/q_model.pt")
    parser.add_argument("--goal-layers", type=int, default=2)
    parser.add_argument("--head-layers", type=int, default=4)
    parser.add_argument("--chain", nargs="+", default=DEFAULT_CHAIN,
                        help=f"ordered goal names; valid: {GOAL_NAMES}")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--temp", type=float, default=0.01,
                        help="softmax temperature for the softmax readout")
    parser.add_argument("--seed", type=int, default=0,
                        help="base seed; episode i uses seed+i")
    parser.add_argument("--budget-mult", type=float, default=1.0,
                        help="scale per-leg step budget (>1 relaxes the anti-circling cap)")
    parser.add_argument("--readouts", nargs="+", default=["greedy", "softmax"],
                        choices=["greedy", "softmax"])
    args = parser.parse_args()

    bad = [n for n in args.chain if n not in GOAL_NAMES]
    if bad:
        parser.error(f"unknown goal name(s) {bad}; valid: {GOAL_NAMES}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = gym.make(
        "HomeBot2D-V1",
        render_mode="rgb_array",
        action_mode="discrete",
        obs_resolution=(96, 96),
        n_trash=2,
        max_steps=20000,   # never let env truncation cut a chain; per-leg budget is the timeout
        map_name="default",
        random_start=True,
    )
    n_actions = env.action_space.n  # type: ignore[union-attr]
    model = load_q_model(args.checkpoint, n_actions, device,
                         goal_layers=args.goal_layers, head_layers=args.head_layers)

    print(f"\nchain: {args.chain}")
    print(f"checkpoint: {args.checkpoint} (goal_layers={args.goal_layers}, "
          f"head_layers={args.head_layers}) | reach<= {GOAL_THRESHOLD}px")

    summary = {}
    for readout in args.readouts:
        episodes_results = [
            run_chain(model, env, args.chain, device, readout, args.temp,
                      seed=args.seed + i, budget_mult=args.budget_mult)
            for i in range(args.episodes)
        ]
        summary[readout] = _print_readout(readout, episodes_results, args.chain)

    print("\n=== Summary (chain completion / full-chain) ===")
    for readout, (comp, full) in summary.items():
        print(f"{readout:<8} {comp:.0f}% / {full:.0f}%")


if __name__ == "__main__":
    main()
