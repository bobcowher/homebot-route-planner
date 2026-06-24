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
from motion import MotionState
from policy import softmax_rel_probs, decode_macro
from task_chain import DEFAULT_CHAIN, resolve_goal, world_state, leg_succeeded

VALID_NAMES = set(GOAL_NAMES) | {"go_to_human"}

# go_to_door lands on a walk-on doorway tile, so "reached" must mean the robot is
# actually AT the doorway — not the 79px solid-fixture reach, which scores a hit
# 2+ tiles short (outside the hallway). Matches the env's tightened package-pickup
# radius: robot.RADIUS(15) + tile_size(32) * _DOOR_RANGE(1.0) = 47px. Solid
# fixtures (fridge/recliner) keep GOAL_THRESHOLD — the robot can't stand on them.
# Per-target STOPPING radius: match the env's own interaction radius so the robot
# is driven close enough for the env interaction (pickup/collect) to actually
# fire. robot.RADIUS(15) + tile_size(32) * RANGE:
#   door  package pickup  _DOOR_RANGE 1.0 -> 47px
#   trash floor collect   _TRASH_RANGE 0.5 -> 31px   (default 79px stopped 1.5
#         tiles short, so trash was never collected -- see leg_succeeded)
# Fixtures (fridge/recliner) interact at _FIXTURE_RANGE 2.0 = 79px = GOAL_THRESHOLD,
# so they need no override.
DOOR_REACH = 47.0
TRASH_REACH = 31.0
REACH_OVERRIDE = {"go_to_door": DOOR_REACH, "collect_trash": TRASH_REACH}


def _select_action(model, obs, goal_xy, robot, device, readout, temp, motion):
    """One greedy/softmax action from the current obs + pose toward goal_xy."""
    with torch.no_grad():
        obs_t = obs.unsqueeze(0).float().to(device) / 255.0
        goal_vec = world_coords(robot.x, robot.y, goal_xy[0], goal_xy[1])
        goal_t = torch.as_tensor(goal_vec, dtype=torch.float32,
                                 device=device).unsqueeze(0)
        motion_t = None
        if getattr(model, "use_motion", False):
            motion_t = torch.as_tensor(motion, dtype=torch.float32, device=device).unsqueeze(0)
        q = model(obs_t, goal_t, motion_t).squeeze(0)
        if readout == "softmax":
            # Absolute temperature (correct for a soft-Q model: temp == alpha).
            probs = F.softmax(q / temp, dim=0)
            return int(torch.multinomial(probs, 1).item())
        if readout == "softmax_rel":
            # Scale-invariant softmax (shared with the training rollout policy in
            # agent.select_action, so train==deploy). temp is a unitless fraction of
            # the per-state Q spread; breaks limit cycles without a magnitude magic
            # number. See policy.softmax_rel_probs.
            probs = softmax_rel_probs(q, temp)
            return int(torch.multinomial(probs, 1).item())
        return int(q.argmax().item())


def run_leg(model, env, base, obs, goal_xy, budget, device, readout, temp, ms, reach):
    """Drive the navigator toward goal_xy until reached or budget exhausted.
    ms is the per-episode MotionState (persists across legs); reach is the
    per-leg reach radius in px (tighter for the walk-on door than for fixtures).

    Returns (reached, steps, obs, positions). obs is threaded back out so the
    next leg continues from the live observation without an env reset. positions
    is the per-step (x, y) trace (incl. the start pose) for the spin metric.
    """
    robot = base._robot
    positions = [(robot.x, robot.y)]
    macro_h = getattr(model, "macro_h", 1)
    n_base = getattr(model, "n_base", env.action_space.n)
    steps = 0
    while steps < budget:
        motion = ms.vec(robot.x, robot.y)
        idx = _select_action(model, obs, goal_xy, robot, device, readout, temp, motion)
        # Execute the decoded macro open-loop (macro_h=1 -> a single action),
        # checking reach after every base step so a mid-macro arrival stops.
        actions = decode_macro(idx, macro_h, n_base)
        for a in actions:
            ms.commit(robot.x, robot.y, a)
            obs = process_observation(env.step(a)[0])
            positions.append((robot.x, robot.y))
            steps += 1
            if distance(robot.x, robot.y, goal_xy[0], goal_xy[1]) <= reach:
                return True, steps, obs, positions
            if steps >= budget:
                break
    return False, steps, obs, positions


def run_chain(model, env, chain, device, readout, temp, seed, budget_mult=1.0):
    """Reset once, walk the chain leg-by-leg. Pose persists across legs; a failed
    leg does NOT abort the chain (we continue so we can see where it breaks).

    budget_mult scales the per-leg step budget (1.0 = goal_geometry's default
    anti-circling budget; larger relaxes it toward the training eval's full
    episode length for comparability).

    Returns list of (name, reached, steps, positions) per leg.
    """
    base = env.unwrapped
    raw_obs, _ = env.reset(seed=seed)
    obs = process_observation(raw_obs)
    robot = base._robot
    ms = MotionState(env.action_space.n, getattr(model, "motion_window", 1))  # motion persists across legs

    # Resolve every leg's target coordinate up front (the static orchestrated list
    # is coords fixed at plan time). Must happen before stepping: the robot picks
    # up trash incidentally while traversing earlier legs, which would empty
    # trash_positions and make a later collect_trash leg unresolvable.
    targets = [(name, resolve_goal(base, name)) for name in chain]

    results = []
    for name, (gx, gy) in targets:
        budget = max(1, int(eval_step_budget(distance(robot.x, robot.y, gx, gy)) * budget_mult))
        reach = REACH_OVERRIDE.get(name, GOAL_THRESHOLD)
        before = world_state(base)
        arrived, steps, obs, positions = run_leg(model, env, base, obs, (gx, gy),
                                                 budget, device, readout, temp, ms, reach)
        # Honest score: the leg counts only if the TASK actually happened (state
        # delta), not merely that the robot got near the coordinate.
        reached = leg_succeeded(name, before, world_state(base), arrived)
        results.append((name, reached, steps, positions))
    return results


def _print_readout(label, episodes_results, chain):
    """episodes_results: list (per episode) of list of (name, reached, steps)."""
    n_ep = len(episodes_results)
    print(f"\n=== readout: {label} | {n_ep} episode(s) ===")

    # Per-leg reach counts indexed by POSITION (a name like go_to_human can
    # repeat in the chain, so name-keying would collapse the return trips).
    n_legs = len(chain)
    per_leg_reached = [0] * n_legs
    scores = []              # legs reached this episode (0..n_legs)
    full_chain = 0           # episodes with every leg reached

    for ep_i, legs in enumerate(episodes_results):
        reached_here = sum(1 for _, r, *_ in legs if r)
        scores.append(reached_here)
        if reached_here == n_legs:
            full_chain += 1
        for i, (_, r, *_) in enumerate(legs):
            if r:
                per_leg_reached[i] += 1
        if n_ep <= 3:  # detailed leg table only for small runs
            print(f"  episode {ep_i}: score {reached_here}/{n_legs}")
            for name, r, steps, _ in legs:
                mark = "reached" if r else "TIMEOUT"
                print(f"    {name:<16} {mark:<8} steps={steps}")

    print("  per-leg reach rate (by position):")
    for i, name in enumerate(chain):
        print(f"    {i+1}. {name:<16} {per_leg_reached[i]}/{n_ep} "
              f"= {100.0 * per_leg_reached[i] / n_ep:.0f}%")
    mean_score = sum(scores) / n_ep
    print(f"  MEAN SCORE: {mean_score:.2f} / {n_legs}")
    print(f"  full-chain (all {n_legs}): {full_chain}/{n_ep} "
          f"= {100.0 * full_chain / n_ep:.0f}%")
    return mean_score, 100.0 * full_chain / n_ep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/q_model.pt")
    parser.add_argument("--goal-layers", type=int, default=2)
    parser.add_argument("--head-layers", type=int, default=4)
    parser.add_argument("--chain", nargs="+", default=DEFAULT_CHAIN,
                        help=f"ordered goal names; valid: {sorted(VALID_NAMES)}")
    parser.add_argument("--head-norm", action="store_true",
                        help="checkpoint was trained with LayerNorm head")
    parser.add_argument("--use-motion", action="store_true",
                        help="checkpoint was trained with the motion input")
    parser.add_argument("--motion-window", type=int, default=1,
                        help="windowed net-displacement horizon the checkpoint was "
                             "trained with (1 = original velocity-only motion)")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--temp", type=float, default=0.01,
                        help="softmax temperature for the softmax readout")
    parser.add_argument("--seed", type=int, default=0,
                        help="base seed; episode i uses seed+i")
    parser.add_argument("--budget-mult", type=float, default=1.0,
                        help="scale per-leg step budget (>1 relaxes the anti-circling cap)")
    parser.add_argument("--readouts", nargs="+", default=["greedy", "softmax"],
                        choices=["greedy", "softmax", "softmax_rel"])
    args = parser.parse_args()

    bad = [n for n in args.chain if n not in VALID_NAMES]
    if bad:
        parser.error(f"unknown goal name(s) {bad}; valid: {sorted(VALID_NAMES)}")

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
                         goal_layers=args.goal_layers, head_layers=args.head_layers,
                         head_norm=args.head_norm, use_motion=args.use_motion,
                         motion_window=args.motion_window)

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

    n_legs = len(args.chain)
    print(f"\n=== Summary (mean score / {n_legs} | full-chain%) ===")
    for readout, (score, full) in summary.items():
        print(f"{readout:<8} {score:.2f} / {n_legs}   {full:.0f}%")


if __name__ == "__main__":
    main()
