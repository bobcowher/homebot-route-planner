"""Smoke-test random-tile goal training.

Verifies: _reset_goal samples varied valid floor tiles, the doorway coord is in
the candidate set (the coverage gap chained_eval found), the env's internal goal
is synced, and a short rollout steps without error.

    conda run -n sac-homebot python scripts/smoke_random_goals.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np

import homebot  # noqa: F401
from agent import Agent

env = gym.make(
    "HomeBot2D-Goal-V1", render_mode="rgb_array", action_mode="discrete",
    obs_resolution=(96, 96), n_trash=2, max_steps=1000, map_name="default",
    goals=["collect_trash"], random_start=True,
)
agent = Agent(env=env, max_buffer_size=1000, goal_layers=2, head_layers=4,
              random_goal_tiles=True)
base = env.unwrapped

# Doorway coverage: the (23,9)/(23,10) threshold tiles must be valid floor tiles.
tiles = base._map.valid_floor_tiles()
assert (23, 9) in tiles and (23, 10) in tiles, "doorway tiles missing from candidates"
door_px = base._map.tile_to_pixel(23, 9)
print(f"valid floor tiles: {len(tiles)} | doorway (23,9) -> {door_px} present OK")

# _reset_goal returns varied goals, synced to the env, within map bounds.
raw_obs, _ = env.reset()
goals = []
for _ in range(30):
    env.reset()
    g = agent._reset_goal(base, raw_obs["desired_goal"])
    assert np.allclose(g, base._desired_goal), "env _desired_goal not synced to sampled goal"
    assert 0 <= g[0] <= base._map.pixel_width and 0 <= g[1] <= base._map.pixel_height
    goals.append(tuple(g))
assert len(set(goals)) > 5, f"goals not varied: {set(goals)}"
print(f"sampled {len(set(goals))} distinct goals over 30 resets OK")

# Short rollout drives without error and rewards against the overridden goal.
raw_obs, _ = env.reset()
obs = agent.process_observation(raw_obs["observation"])
goal = agent._reset_goal(base, raw_obs["desired_goal"])
r = base._robot
for _ in range(50):
    goal_vec = np.array([r.x, r.y, goal[0], goal[1]], dtype=np.float32)
    action = agent.select_action(obs, goal_vec)
    raw_next, reward, term, trunc, _ = env.step(action)
    obs = agent.process_observation(raw_next["observation"])
    if term or trunc:
        break
print("50-step rollout with random-tile goal OK")
print("RANDOM-GOAL SMOKE OK")
