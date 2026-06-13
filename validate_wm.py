# validate_wm.py
"""Validate a trained HomeBot world-model checkpoint.

(1) Latent trash-visibility: collect diversified frames (teleport robot to random
    floor tiles), run the frozen WM encoder + detection head, and report the
    detection localization hit-rate vs the prior floor (reuses the probe's logic).
(2) Policy success: greedy rollouts, report full-clear rate on the trash task.

Usage:
    python3 validate_wm.py --episodes 100
"""
import argparse
import numpy as np
import torch
import cv2
import gymnasium as gym
import homebot  # noqa: F401
from agent import Agent
from goal_labels import label_rows
from models.detection_head import OBS, GRID


def make_env(max_steps=1000):
    for env_id in ("HomeBot2D-v1", "HomeBot2D-V1"):
        try:
            return gym.make(env_id, render_mode="rgb_array", action_mode="continuous",
                            obs_resolution=(OBS, OBS), n_trash=2, max_steps=max_steps,
                            map_name="default", goals=["trash"])
        except gym.error.Error:
            continue
    raise RuntimeError("No HomeBot2D env id registered")


@torch.no_grad()
def latent_trash_visibility(agent, env, n_frames=4000):
    base = env.unwrapped
    rng = np.random.default_rng(0)
    floor = base._map.valid_floor_tiles()
    env.reset()
    hits = total = 0
    for i in range(n_frames):
        if i % 40 == 0:
            env.reset()
        tx, ty = floor[int(rng.integers(len(floor)))]
        base._robot.x, base._robot.y = base._map.tile_to_pixel(tx, ty)
        frame = cv2.resize(base._get_obs(), (OBS, OBS), interpolation=cv2.INTER_NEAREST)
        rows = label_rows(base)
        true = [(int(r[2]) * GRID // OBS, int(r[1]) * GRID // OBS) for r in rows if r[0] >= 0]
        if not true:
            continue
        total += 1
        obs_t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(agent.device) / 255.0
        embed, _, _ = agent.world_model.encode(obs_t)
        hm = torch.sigmoid(agent.world_model.detection_head(embed.squeeze(1)))[0, 0].cpu().numpy()
        py, px = np.unravel_index(int(hm.argmax()), hm.shape)
        if any(abs(py - gy) <= 1 and abs(px - gx) <= 1 for gy, gx in true):
            hits += 1
    print(f"latent trash-visibility hit-rate: {hits}/{total} = {100 * hits / max(total, 1):.0f}%")


def policy_success(agent, env, episodes=100):
    full = 0
    for ep in range(episodes):
        obs, _ = env.reset()
        obs = agent.process_observation(obs)
        done, ep_r = False, 0.0
        while not done:
            with torch.no_grad():
                obs_t = obs.unsqueeze(0).float().to(agent.device) / 255.0
                embed, _, _ = agent.world_model.encode(obs_t)
                action = agent.select_action(embed.squeeze(1), evaluate=True)
            nxt, r, term, trunc, _ = env.step(action)
            obs = agent.process_observation(nxt)
            done = term or trunc
            ep_r += float(r)
        full += int(ep_r >= 2)
    print(f"policy full-clear rate: {full}/{episodes} = {100 * full / episodes:.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    args = ap.parse_args()
    env = make_env()
    agent = Agent(env=env, max_buffer_size=1000)
    agent.load()  # loads world_model/actor/critic from checkpoints/
    agent.actor.eval()
    latent_trash_visibility(agent, env)
    policy_success(agent, env, args.episodes)
    env.close()


if __name__ == "__main__":
    main()
