# tests/test_integration.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import gymnasium as gym
import homebot  # noqa: F401
from agent import Agent


def test_three_episode_train_runs_without_error():
    env = gym.make(
        "HomeBot2D-Goal-V1",
        render_mode="rgb_array",
        action_mode="discrete",
        obs_resolution=(96, 96),
        n_trash=2,
        max_steps=50,
        map_name="default",
        goals=["collect_trash"],
    )
    agent = Agent(env=env, max_buffer_size=10000)
    agent.train(episodes=3, batch_size=64)
    env.close()
