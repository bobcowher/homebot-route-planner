import numpy as np
from motion import motion_dim_continuous, make_motion_continuous, MotionStateContinuous


def test_motion_dim_window_1():
    assert motion_dim_continuous(window=1) == 4  # [last_lin, last_ang, dx, dy]


def test_motion_dim_window_gt_1():
    assert motion_dim_continuous(window=8) == 6  # + net_dx, net_dy


def test_make_motion_continuous_first_step_no_last_action():
    m = make_motion_continuous(last_action=None, dx=4.0, dy=0.0, step=4.0, window=1)
    assert m.shape == (4,)
    assert np.allclose(m, [0.0, 0.0, 1.0, 0.0])


def test_make_motion_continuous_with_last_action():
    m = make_motion_continuous(last_action=np.array([0.5, -1.0]), dx=2.0, dy=2.0, step=4.0, window=1)
    assert np.allclose(m, [0.5, -1.0, 0.5, 0.5])


def test_make_motion_continuous_windowed():
    m = make_motion_continuous(last_action=np.array([1.0, 0.0]), dx=4.0, dy=0.0,
                                net_dx=8.0, net_dy=0.0, step=4.0, window=8)
    assert m.shape == (6,)
    assert np.allclose(m, [1.0, 0.0, 1.0, 0.0, 0.25, 0.0])


def test_motion_state_continuous_first_vec_is_zero_motion():
    ms = MotionStateContinuous(window=1)
    m = ms.vec(100.0, 100.0)
    assert np.allclose(m, [0.0, 0.0, 0.0, 0.0])


def test_motion_state_continuous_tracks_displacement_and_last_action():
    ms = MotionStateContinuous(window=1)
    ms.vec(100.0, 100.0)
    ms.commit(100.0, 100.0, np.array([1.0, 0.0]))
    m = ms.vec(104.0, 100.0)
    assert np.allclose(m, [1.0, 0.0, 1.0, 0.0])
