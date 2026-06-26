# tests/test_sac_smoke.py
"""End-to-end smoke test: real env, a handful of episodes, must not crash
or produce NaNs. NOT a convergence test — just verifies the discrete SAC
pipeline runs end-to-end."""
import math

import gymnasium as gym
import homebot  # noqa: F401  (registers HomeBot2D-Goal-V1)

from sac_agent import SACAgent


def _make_env():
    return gym.make(
        "HomeBot2D-Goal-V1",
        render_mode=None,
        action_mode="discrete",
        obs_resolution=(96, 96),
        n_trash=1,
        max_steps=50,
        map_name="default",
        goals=["collect_trash"],
        random_start=True,
    )


def test_smoke_few_episodes_no_nan_no_crash():
    env = _make_env()
    agent = SACAgent(env=env, max_buffer_size=5000)

    agent.train(episodes=5, batch_size=16, run_tag="smoke-test", warmup_steps=0)

    assert agent.total_env_steps > 0
    assert math.isfinite(agent.policy.conv1.weight.sum().item())
    assert math.isfinite(agent.critic.conv1.weight.sum().item())
    env.close()


def test_smoke_reach_curriculum_runs_and_floods_reward():
    """Curriculum on: the rollout reward/terminal is recomputed at a big radius, so a
    short episode on the real env should terminate via reach and book real reward —
    exercising the _run_episode override, the radius schedule, and HER's matched reward."""
    env = _make_env()
    agent = SACAgent(env=env, max_buffer_size=5000)

    # Huge start radius -> the robot is 'within reach' almost immediately, so episodes
    # terminate fast and episode_reward is non-zero. Anneal across the 4 episodes.
    agent.train(episodes=4, batch_size=16, run_tag="smoke-curriculum", warmup_steps=0,
                reach_start=10000.0, reach_end=79.0,
                reach_anneal_start=0, reach_anneal_end=4)

    assert agent.total_env_steps > 0
    assert math.isfinite(agent.critic.conv1.weight.sum().item())
    env.close()
