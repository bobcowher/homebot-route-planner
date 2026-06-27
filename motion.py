"""Per-step motion feature for the discrete SAC policy (anti-oscillation cue)."""
import numpy as np

from goal_geometry import ROBOT_STEP_PX


class MotionStateDiscrete:
    """[dx/step, dy/step, 0.0, 0.0] — 4 dims, matches the replay buffer MOTION_DIM.

    Tracks per-step displacement only; the discrete action index is not encoded here
    because DiscreteQNet evaluates all actions simultaneously.

    Usage each step at robot pose (x, y):
        motion = ms.vec(x, y)
        action = agent.greedy_critic_action(obs, goal, motion)
        ms.commit(x, y, action)
        env.step(action)
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
