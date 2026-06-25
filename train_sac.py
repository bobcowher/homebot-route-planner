# train_sac.py
"""Continuous SAC + HER on the collect_trash leg. Stability-first v1: no
image observation, no n-step, no macro-actions, no goal noise. Mirrors
train.py's collect_trash config (n_trash=1, random_start=True) so this is
a clean A/B against the discrete champion's reference run, modulo algorithm."""
import gymnasium as gym
import homebot  # noqa: F401

from sac_agent import SACAgent
from sac_motion import motion_dim_continuous

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

STATE_DIM = 2 + motion_dim_continuous(window=1)  # ego_vector(2) + motion(4)

agent = SACAgent(env=env, state_dim=STATE_DIM, action_dim=2,
                 max_buffer_size=200000, hidden_dim=128,
                 gamma=0.99, tau=0.005, alpha=0.1, lr=3e-4, motion_window=1)

agent.train(episodes=1800, batch_size=64)
