import gymnasium as gym
import pytest

import homebot  # noqa: F401  (env registration)
from planner.world_model import WorldModel, DEST_TO_ENV


def _env():
    return gym.make("HomeBot2D-V1", render_mode="rgb_array", action_mode="discrete",
                    obs_resolution=(96, 96), n_trash=2, max_steps=20000,
                    map_name="default", random_start=True)


def test_list_destinations_includes_fixtures_and_trash_when_present():
    env = _env(); env.reset(seed=0)
    w = WorldModel(env)
    dests = w.list_destinations()
    assert {"fridge", "human", "door"} <= set(dests)
    assert "trash" in dests  # n_trash=2 -> trash present right after reset


def test_resolve_known_returns_float_pair_unknown_raises():
    env = _env(); env.reset(seed=0)
    w = WorldModel(env)
    xy = w.resolve("fridge")
    assert len(xy) == 2 and all(isinstance(v, float) for v in xy)
    with pytest.raises(KeyError):
        w.resolve("bogus")


def test_state_exposes_carry_and_trash():
    env = _env(); env.reset(seed=0)
    w = WorldModel(env)
    s = w.state()
    assert {"carrying", "trash_remaining", "robot_xy"} <= set(s)
    assert s["trash_remaining"] == 2
