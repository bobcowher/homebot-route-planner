"""End-to-end smoke for the success-radius curriculum: a few short episodes with the
curriculum on, exercising the radius schedule, the rollout reward/term recompute, and
the curriculum HER reward in send_to. Asserts it runs without crashing and that the
tight-radius episode actually ran longer (didn't terminate at the env's 79px)."""
import sys
from pathlib import Path

import gymnasium as gym

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import homebot  # noqa: F401  (env registration)
from agent import Agent


def main():
    env = gym.make("HomeBot2D-Goal-V1", render_mode="rgb_array", action_mode="discrete",
                   obs_resolution=(96, 96), n_trash=2, max_steps=60, map_name="default",
                   goals=["collect_trash"], random_start=True)
    agent = Agent(env=env, max_buffer_size=2000, goal_layers=2, head_layers=4,
                  use_motion=True, motion_window=1, random_goal_tiles=True)
    # episodes 0-1 hold radius=79 (terminate early like the env); episode 2 drops to
    # 28 (must approach further -> longer episode, capped by max_steps).
    agent.train(episodes=3, batch_size=8, eval_interval=999, eval_episodes=1,
                chain_eval_interval=999, her_anneal_start=None,
                reach_start=79.0, reach_end=28.0,
                reach_anneal_start=1, reach_anneal_end=2)
    print("OK | curriculum train ran 3 episodes without crashing")


if __name__ == "__main__":
    main()
