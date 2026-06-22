"""Unit tests for the success-radius curriculum primitives (no env import)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goal_geometry import reach_reward, reach_radius_at


def test_reach_reward_within_and_outside():
    g = np.array([100.0, 100.0], dtype=np.float32)
    assert reach_reward(np.array([110.0, 100.0]), g, 31.0) == 1.0   # 10px < 31
    assert reach_reward(np.array([150.0, 100.0]), g, 31.0) == 0.0   # 50px > 31
    # Boundary is inclusive (<=).
    assert reach_reward(np.array([131.0, 100.0]), g, 31.0) == 1.0


def test_reach_reward_radius_shrinks_acceptance():
    g = np.array([0.0, 0.0], dtype=np.float32)
    p = np.array([40.0, 0.0], dtype=np.float32)            # 40px away
    assert reach_reward(p, g, 79.0) == 1.0                  # inside loose bar
    assert reach_reward(p, g, 28.0) == 0.0                  # outside tight bar


def test_reach_reward_batched():
    g = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    a = np.array([[10.0, 0.0], [50.0, 0.0]], dtype=np.float32)
    out = reach_reward(a, g, 31.0)
    assert out.shape == (2,)
    assert out[0] == 1.0 and out[1] == 0.0


def test_reach_radius_schedule_endpoints_and_monotone():
    # hold start before anneal_start
    assert reach_radius_at(0, 79.0, 28.0, 100, 600) == 79.0
    assert reach_radius_at(100, 79.0, 28.0, 100, 600) == 79.0
    # hold end after anneal_end
    assert reach_radius_at(600, 79.0, 28.0, 100, 600) == 28.0
    assert reach_radius_at(1800, 79.0, 28.0, 100, 600) == 28.0
    # midpoint is the average
    mid = reach_radius_at(350, 79.0, 28.0, 100, 600)
    assert abs(mid - (79.0 + 28.0) / 2) < 1e-5
    # monotonically non-increasing across the anneal
    vals = [reach_radius_at(e, 79.0, 28.0, 100, 600) for e in range(0, 700, 50)]
    assert all(b <= a + 1e-9 for a, b in zip(vals, vals[1:]))
