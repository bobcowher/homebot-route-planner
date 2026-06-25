import numpy as np
import torch
import gymnasium as gym
from sac_model import Policy, Critic


def _action_space():
    return gym.spaces.Box(low=np.array([-1., -1.], dtype=np.float32),
                          high=np.array([1., 1.], dtype=np.float32), dtype=np.float32)


def _inputs(batch=8):
    image  = torch.rand(batch, 3, 96, 96)
    goal   = torch.rand(batch, 2)
    motion = torch.rand(batch, 4)
    action = torch.rand(batch, 2)
    return image, goal, motion, action


def test_critic_forward_shapes():
    critic = Critic(action_dim=2)
    image, goal, motion, action = _inputs()
    q1, q2 = critic(image, goal, motion, action)
    assert q1.shape == (8, 1)
    assert q2.shape == (8, 1)


def test_policy_sample_shapes_and_action_bounds():
    policy = Policy(action_dim=2, action_space=_action_space())
    image, goal, motion, _ = _inputs()
    action, log_prob, mean = policy.sample(image, goal, motion)
    assert action.shape == (8, 2)
    assert log_prob.shape == (8, 1)
    assert mean.shape == (8, 2)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)


def test_policy_sample_is_stochastic():
    policy = Policy(action_dim=2, action_space=_action_space())
    image, goal, motion, _ = _inputs(batch=1)
    a1, _, _ = policy.sample(image, goal, motion)
    a2, _, _ = policy.sample(image, goal, motion)
    assert not torch.allclose(a1, a2)
