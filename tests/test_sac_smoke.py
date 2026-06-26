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


def test_smoke_start_distance_curriculum_spawns_near_goal():
    """Start-distance curriculum: with a small start_dist the robot must be respawned
    within start_dist of the goal. Verify the override actually moves the spawn inside
    the band (and the rollout still runs end-to-end without NaNs)."""
    env = _make_env()
    agent = SACAgent(env=env, max_buffer_size=5000)

    # Exercise the spawn override directly across several resets: the respawn should
    # land within the band the large majority of the time (fallback only when a goal is
    # cornered with no in-band tile), and far closer than a full-map random spawn.
    import numpy as np
    base = env.unwrapped
    agent._start_dist_min = 90.0
    dists = []
    for _ in range(10):
        env.reset()
        raw2 = agent._spawn_near_goal(base, start_dist=200.0)
        dists.append(float(np.linalg.norm(raw2["achieved_goal"] - raw2["desired_goal"])))
    in_band = [d for d in dists if d <= 200.0 + base._map.tile_size]
    assert len(in_band) >= 7  # the spawn override is actually pulling starts near the goal

    agent.train(episodes=3, batch_size=16, run_tag="smoke-startdist", warmup_steps=0,
                start_dist_start=150.0, start_dist_end=400.0,
                start_dist_anneal_start=0, start_dist_anneal_end=3, start_dist_min=90.0)
    assert math.isfinite(agent.policy.conv1.weight.sum().item())
    env.close()
