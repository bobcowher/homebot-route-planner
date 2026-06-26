"""Continuous-action analog of motion.py's anti-oscillation features.

motion.py's last-action term is a one-hot sized by the discrete action_dim,
which doesn't apply to a continuous [linear, angular] action. This module
keeps the same windowed-net-displacement idea (still a valid spin signal for
a heading-controlled policy) but represents the previous action as its raw
2-vector instead of a one-hot index.
"""
from collections import deque

import numpy as np

from goal_geometry import ROBOT_STEP_PX


def motion_dim_continuous(window: int = 1) -> int:
    """[last_linear, last_angular, dx, dy] plus [net_dx, net_dy] when window > 1."""
    return 2 + 2 + (2 if window > 1 else 0)


def make_motion_continuous(last_action, dx, dy, net_dx=0.0, net_dy=0.0,
                            step=ROBOT_STEP_PX, window=1):
    """[last_action(2) | dx/step | dy/step | net_dx/(W*step) | net_dy/(W*step)].

    last_action None -> zeros (episode start). Velocity normalized by the max
    per-step speed so it sits in ~[-1, 1], same convention as motion.py."""
    m = np.zeros(motion_dim_continuous(window), dtype=np.float32)
    if last_action is not None:
        m[0] = float(last_action[0])
        m[1] = float(last_action[1])
    m[2] = dx / step
    m[3] = dy / step
    if window > 1:
        m[4] = net_dx / (window * step)
        m[5] = net_dy / (window * step)
    return m


class MotionStateContinuous:
    """Per-rollout tracker, continuous-action analog of motion.MotionState.

    Usage each step, at robot pose (x, y):
        motion = ms.vec(x, y)
        action = policy.select_action(state_with(motion))
        ms.commit(x, y, action)
        env.step(action)
    """

    def __init__(self, window: int = 1):
        self.window = window
        self.reset()

    def reset(self):
        self.last_action = None
        self.prev = None
        self.history = deque(maxlen=max(1, self.window))

    def vec(self, x, y):
        if self.prev is None:
            dx = dy = 0.0
        else:
            dx, dy = x - self.prev[0], y - self.prev[1]
        if self.window > 1 and self.history:
            ox, oy = self.history[0]
            net_dx, net_dy = x - ox, y - oy
        else:
            net_dx = net_dy = 0.0
        return make_motion_continuous(self.last_action, dx, dy, net_dx, net_dy,
                                      window=self.window)

    def commit(self, x, y, action):
        self.history.append((x, y))
        self.prev = (x, y)
        self.last_action = action


class MotionStateDiscrete:
    """[dx/step, dy/step, 0.0, 0.0] — 4 dims, matches buffer MOTION_DIM.

    Tracks per-step displacement only; discrete action index is not encoded
    here because DiscreteQNet evaluates all actions simultaneously.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.prev = None

    def vec(self, x: float, y: float) -> np.ndarray:
        if self.prev is None:
            dx, dy = 0.0, 0.0
        else:
            dx = (x - self.prev[0]) / ROBOT_STEP_PX
            dy = (y - self.prev[1]) / ROBOT_STEP_PX
        return np.array([dx, dy, 0.0, 0.0], dtype=np.float32)

    def commit(self, x: float, y: float, action: int):  # noqa: ARG002
        self.prev = (x, y)
