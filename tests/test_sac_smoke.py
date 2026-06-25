# tests/test_sac_smoke.py
"""End-to-end smoke test: real env, a handful of episodes, must not crash
or produce NaNs. This is NOT a convergence test -- it's the pipeline-runs
check from the design spec, mirroring this repo's existing HER smoke tests."""
import math

import gymnasium as gym
import homebot  # noqa: F401  (registers HomeBot2D-Goal-V1)
import numpy as np

from sac_agent import SACAgent
from sac_motion import motion_dim_continuous
from goal_geometry import ego_vector


def _make_env():
    return gym.make(
        "HomeBot2D-Goal-V1",
        render_mode=None,
        action_mode="continuous",
        obs_resolution=(96, 96),
        n_trash=1,
        max_steps=50,  # short episodes -- this test just needs the pipe to run
        map_name="default",
        goals=["collect_trash"],
        random_start=True,
    )


def test_smoke_few_episodes_no_nan_no_crash():
    env = _make_env()
    state_dim = 2 + motion_dim_continuous(window=1)  # ego_vector(2) + motion(4)
    agent = SACAgent(env=env, state_dim=state_dim, action_dim=2,
                     max_buffer_size=5000, hidden_dim=32)

    agent.train(episodes=5, batch_size=16, run_tag="smoke-test")

    assert agent.total_env_steps > 0
    assert math.isfinite(agent.policy.linear1.weight.sum().item())
    assert math.isfinite(agent.critic.linear1.weight.sum().item())
    env.close()


def test_smoke_state_dim_matches_what_build_state_produces():
    env = _make_env()
    state_dim = 2 + motion_dim_continuous(window=1)
    agent = SACAgent(env=env, state_dim=state_dim, action_dim=2,
                     max_buffer_size=5000, hidden_dim=32)

    raw_obs, _ = env.reset()
    base = env.unwrapped
    r = base._robot
    motion = np.zeros(motion_dim_continuous(window=1), dtype=np.float32)
    state = agent._build_state(r.x, r.y, r.angle,
                               raw_obs["desired_goal"][0], raw_obs["desired_goal"][1], motion)
    assert state.shape == (state_dim,)
    env.close()
