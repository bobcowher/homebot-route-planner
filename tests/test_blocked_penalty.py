"""Coverage for the blocked-move (wall-pin) penalty in episode_buffer.

The penalty must fire on a true pin (zero displacement), stay off for a normal
move and for a one-axis wall-slide, and apply to BOTH the original and the
hindsight-relabeled copies (goal-independent).
"""
import numpy as np

from episode_buffer import (EpisodeBuffer, Transition, _blocked_penalty,
                            BLOCKED_PENALTY)


def _t(prev, nxt, reward=0.0, done=False):
    return Transition(
        obs=None, action=0, reward=reward, next_obs=None, done=done,
        achieved_prev=np.array(prev, dtype=np.float32),
        achieved_next=np.array(nxt, dtype=np.float32),
        heading_prev=0.0, heading_next=0.0,
    )


def test_pin_is_penalized():
    assert _blocked_penalty(_t((10, 10), (10, 10))) == BLOCKED_PENALTY


def test_full_move_not_penalized():
    assert _blocked_penalty(_t((10, 10), (14, 10))) == 0.0


def test_wall_slide_not_penalized():
    # diagonal blocked → slide one axis ~2.83px; that's progress, not a pin
    assert _blocked_penalty(_t((10, 10), (12.83, 10))) == 0.0


class _StubBuf:
    def __init__(self):
        self.rewards = []

    def store_transition(self, s, a, r, ns, d, g, ng, motion=None, next_motion=None):
        self.rewards.append(float(r))


def _reach_79(achieved, desired, info):
    return (np.linalg.norm(np.asarray(achieved) - np.asarray(desired), axis=-1)
            <= 79.0).astype(np.float32)


def test_penalty_applied_in_both_passes():
    eb = EpisodeBuffer()
    # t0: a pin (no displacement). t1: a real move that reaches the goal.
    eb.store(None, 0, 0.0, None, False, np.array([10, 10], np.float32),
             np.array([10, 10], np.float32))
    eb.store(None, 0, 1.0, None, True, np.array([10, 10], np.float32),
             np.array([100, 100], np.float32))
    buf = _StubBuf()
    eb.send_to(buf, desired_goal=np.array([100, 100], np.float32),
               compute_reward=_reach_79, k=2)

    # Original pass: the pin transition's stored reward is 0.0 + penalty.
    assert any(abs(r - BLOCKED_PENALTY) < 1e-6 for r in buf.rewards), buf.rewards
    # No stored reward should be a bare 0.0 for the pin (penalty always applied).
    assert 0.0 not in buf.rewards
