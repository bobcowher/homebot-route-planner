import numpy as np
import torch
import gymnasium as gym
from sac_model import Policy, Critic


def _action_space():
    return gym.spaces.Box(low=np.array([-1., -1.], dtype=np.float32),
                          high=np.array([1., 1.], dtype=np.float32), dtype=np.float32)


def test_critic_forward_shapes():
    critic = Critic(num_inputs=6, num_actions=2, hidden_dim=128)
    state = torch.randn(8, 6)
    action = torch.randn(8, 2)
    q1, q2 = critic(state, action)
    assert q1.shape == (8, 1)
    assert q2.shape == (8, 1)


def test_policy_sample_shapes_and_action_bounds():
    policy = Policy(num_inputs=6, num_actions=2, hidden_dim=128, action_space=_action_space())
    state = torch.randn(8, 6)
    action, log_prob, mean = policy.sample(state)
    assert action.shape == (8, 2)
    assert log_prob.shape == (8, 1)
    assert mean.shape == (8, 2)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)


def test_policy_sample_is_stochastic():
    policy = Policy(num_inputs=6, num_actions=2, hidden_dim=128, action_space=_action_space())
    state = torch.randn(1, 6)
    a1, _, _ = policy.sample(state)
    a2, _, _ = policy.sample(state)
    assert not torch.allclose(a1, a2)
