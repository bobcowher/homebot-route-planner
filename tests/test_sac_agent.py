import numpy as np
import gymnasium as gym
import torch
from sac_agent import SACAgent


class FakeEnv:
    """Minimal stand-in -- only what SACAgent's constructor touches."""
    action_space = gym.spaces.Box(low=np.array([-1., -1.], dtype=np.float32),
                                  high=np.array([1., 1.], dtype=np.float32), dtype=np.float32)


def test_select_action_within_bounds():
    agent = SACAgent(env=FakeEnv(), state_dim=6, action_dim=2, max_buffer_size=1000, hidden_dim=32)
    state = np.random.randn(6).astype(np.float32)
    action = agent.select_action(state)
    assert action.shape == (2,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_select_action_evaluate_is_deterministic():
    agent = SACAgent(env=FakeEnv(), state_dim=6, action_dim=2, max_buffer_size=1000, hidden_dim=32)
    state = np.random.randn(6).astype(np.float32)
    a1 = agent.select_action(state, evaluate=True)
    a2 = agent.select_action(state, evaluate=True)
    assert np.allclose(a1, a2)  # evaluate path returns the policy mean, not a sample


def test_update_parameters_runs_without_nan_and_returns_three_floats():
    agent = SACAgent(env=FakeEnv(), state_dim=6, action_dim=2, max_buffer_size=1000, hidden_dim=32)
    for _ in range(200):
        s = np.random.randn(6).astype(np.float32)
        a = np.random.uniform(-1, 1, size=2).astype(np.float32)
        ns = np.random.randn(6).astype(np.float32)
        agent.memory.store_transition(s, a, float(np.random.rand()), ns, False)

    critic_loss, policy_loss, mean_q = agent.update_parameters(batch_size=16)
    assert np.isfinite(critic_loss)
    assert np.isfinite(policy_loss)
    assert np.isfinite(mean_q)


def test_update_parameters_changes_policy_weights():
    agent = SACAgent(env=FakeEnv(), state_dim=6, action_dim=2, max_buffer_size=1000, hidden_dim=32)
    for _ in range(200):
        s = np.random.randn(6).astype(np.float32)
        a = np.random.uniform(-1, 1, size=2).astype(np.float32)
        ns = np.random.randn(6).astype(np.float32)
        agent.memory.store_transition(s, a, float(np.random.rand()), ns, False)

    before = agent.policy.linear1.weight.clone().detach()
    agent.update_parameters(batch_size=16)
    after = agent.policy.linear1.weight.clone().detach()
    assert not torch.allclose(before, after)
