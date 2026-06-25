"""Continuous SAC + HER on the collect_trash leg.

CNN-based policy: image (96×96 RGB) + noisy_world_vector goal + motion.
Mirrors train.py's collect_trash config (n_trash=1, random_start=True)
so this is a clean A/B against the discrete champion, modulo algorithm.
"""
import gymnasium as gym
import homebot  # noqa: F401

from sac_agent import SACAgent

env = gym.make(
    "HomeBot2D-Goal-V1",
    render_mode="rgb_array",
    action_mode="continuous",
    obs_resolution=(96, 96),
    n_trash=1,
    max_steps=1000,
    map_name="default",
    goals=["collect_trash"],
    random_start=True,
)

agent = SACAgent(
    env=env,
    action_dim=2,
    max_buffer_size=200000,
    gamma=0.99,
    tau=0.005,
    alpha=0.1,
    lr=3e-4,
    motion_window=1,
    goal_noise_std=30.0,
)

agent.train(episodes=900, batch_size=64, warmup_steps=5000)
