"""Verify the random_start env contract (homebot >= 643898f).

Pure-env check (no torch/Agent): confirms the installed homebot serves
HomeBot2D-Goal-V1 with the random_start param, enforces the >=60px spawn
clearance from the goal, and is seed-deterministic.

Run under the project env:
    conda run -n sac-homebot python scripts/verify_random_start.py
"""
import sys
import math

import gymnasium as gym
import homebot  # noqa: F401  (registers the env)

ENV_ID = "HomeBot2D-Goal-V1"          # single canonical id (renamed capital-V1 at homebot 2ed357a)
CLEARANCE = 60.0                      # 4 * Robot.RADIUS (env's _RANDOM_START_CLEARANCE)
N = 100


def make(**extra):
    return gym.make(
        ENV_ID,
        render_mode="rgb_array",
        action_mode="discrete",
        obs_resolution=(96, 96),
        n_trash=2,
        max_steps=1000,
        map_name="default",
        goals=["collect_trash"],
        **extra,
    )


def robot_xy(env):
    r = env.unwrapped._robot
    return r.x, r.y


def main():
    env = make(random_start=True)
    print(f"env id: {env.spec.id}  (random_start accepted -> homebot is synced)")

    # 1. Clearance: robot spawns >= CLEARANCE from the desired goal every reset.
    min_d = math.inf
    for i in range(N):
        obs, _ = env.reset(seed=i)
        rx, ry = robot_xy(env)
        gx, gy = obs["desired_goal"][0], obs["desired_goal"][1]
        d = math.hypot(rx - gx, ry - gy)
        min_d = min(min_d, d)
    print(f"min robot->goal distance over {N} resets: {min_d:.1f}px (require >= {CLEARANCE})")
    assert min_d >= CLEARANCE - 1e-6, "clearance violated"

    # 2. Determinism: same seed -> same spawn.
    env.reset(seed=4242); a = robot_xy(env)
    env.reset(seed=4242); b = robot_xy(env)
    print(f"determinism: seed 4242 -> {a} then {b}")
    assert a == b, "non-deterministic spawn for fixed seed"

    print("VERIFY OK")


if __name__ == "__main__":
    sys.exit(main())
