"""Smoke: the soft-Q training path runs end-to-end (backup + softmax behavior +
softmax-readout chain metric). A few episodes on the champion config; asserts no
crash and that chain_eval returns a finite score. Not a quality check."""
import sys
from pathlib import Path

import gymnasium as gym

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import homebot  # noqa: F401  (env registration)
from agent import Agent


def main():
    env = gym.make("HomeBot2D-Goal-V1", render_mode="rgb_array",
                   action_mode="discrete", obs_resolution=(96, 96), n_trash=2,
                   max_steps=200, map_name="default", goals=["collect_trash"],
                   random_start=True)
    agent = Agent(env=env, max_buffer_size=5000, goal_layers=2, head_layers=4,
                  use_motion=True, random_goal_tiles=True,
                  soft_q=True, soft_alpha=0.005)
    assert agent.soft_q and abs(agent.soft_alpha - 0.005) < 1e-9

    # behavior policy is the softmax sampler (no exception, valid action)
    raw, _ = env.reset()
    obs = agent.process_observation(raw["observation"])
    from goal_geometry import world_coords
    from motion import MotionState
    ms = MotionState(env.action_space.n)
    a = agent.select_action(obs, world_coords(0, 0, 100, 100), ms.vec(0, 0))
    assert 0 <= a < env.action_space.n

    agent.train(episodes=3, batch_size=16, eval_interval=50,
                eval_episodes=2, chain_eval_interval=2, her_anneal_start=None)

    score, full = agent.chain_eval(n_episodes=2)
    assert score == score, "chain score is NaN"  # NaN != NaN
    print(f"SMOKE OK: soft-Q train+eval clean (chain_score={score:.2f}, full={full:.2f})")


if __name__ == "__main__":
    main()
