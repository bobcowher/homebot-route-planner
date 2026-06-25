# tests/test_sac_smoke.py
"""End-to-end smoke test: real env, a handful of episodes, must not crash
or produce NaNs. This is NOT a convergence test -- it's the pipeline-runs
check from the design spec."""
import math

import gymnasium as gym
import homebot  # noqa: F401  (registers HomeBot2D-Goal-V1)

from sac_agent import SACAgent


def _make_env():
    return gym.make(
        "HomeBot2D-Goal-V1",
        render_mode=None,
        action_mode="continuous",
        obs_resolution=(96, 96),
        n_trash=1,
        max_steps=50,  # short episodes -- test just needs the pipe to run
        map_name="default",
        goals=["collect_trash"],
        random_start=True,
    )


def test_smoke_few_episodes_no_nan_no_crash():
    env = _make_env()
    agent = SACAgent(env=env, action_dim=2, max_buffer_size=5000)

    agent.train(episodes=5, batch_size=16, run_tag="smoke-test", warmup_steps=0)

    assert agent.total_env_steps > 0
    assert math.isfinite(agent.policy.conv1.weight.sum().item())
    assert math.isfinite(agent.critic.conv1.weight.sum().item())
    env.close()
