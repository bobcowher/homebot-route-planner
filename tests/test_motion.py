import numpy as np

from motion import MotionStateDiscrete
from goal_geometry import ROBOT_STEP_PX


def test_first_step_is_zero_displacement():
    ms = MotionStateDiscrete()
    m = ms.vec(100.0, 100.0)
    assert m.shape == (4,)
    assert np.allclose(m, 0.0)


def test_displacement_normalised_by_step():
    ms = MotionStateDiscrete()
    ms.commit(100.0, 100.0, action=2)
    m = ms.vec(100.0 + ROBOT_STEP_PX, 100.0 - ROBOT_STEP_PX)
    assert np.allclose(m, [1.0, -1.0, 0.0, 0.0])


def test_reset_clears_history():
    ms = MotionStateDiscrete()
    ms.commit(50.0, 50.0, action=1)
    ms.reset()
    assert np.allclose(ms.vec(80.0, 80.0), 0.0)
