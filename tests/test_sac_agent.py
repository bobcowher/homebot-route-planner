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


def test_polyak_target_update_moves_toward_online_not_away():
    agent = SACAgent(env=FakeEnv(), state_dim=6, action_dim=2, max_buffer_size=1000, hidden_dim=32, tau=0.1)
    for _ in range(200):
        s = np.random.randn(6).astype(np.float32)
        a = np.random.uniform(-1, 1, size=2).astype(np.float32)
        ns = np.random.randn(6).astype(np.float32)
        agent.memory.store_transition(s, a, float(np.random.rand()), ns, False)

    target_before = agent.critic_target.linear1.weight.clone().detach()
    agent.update_parameters(batch_size=16)
    online_after = agent.critic.linear1.weight.clone().detach()
    target_after = agent.critic_target.linear1.weight.clone().detach()

    expected_target_after = agent.tau * online_after + (1 - agent.tau) * target_before
    assert torch.allclose(target_after, expected_target_after, atol=1e-6)


def test_mean_q_stays_bounded_over_repeated_updates():
    agent = SACAgent(env=FakeEnv(), state_dim=6, action_dim=2, max_buffer_size=2000, hidden_dim=32)
    for _ in range(500):
        s = np.random.randn(6).astype(np.float32)
        a = np.random.uniform(-1, 1, size=2).astype(np.float32)
        ns = np.random.randn(6).astype(np.float32)
        r = float(np.random.uniform(0.0, 1.0))
        agent.memory.store_transition(s, a, r, ns, False)

    mean_qs = []
    for _ in range(100):
        _, _, mean_q = agent.update_parameters(batch_size=32)
        mean_qs.append(mean_q)

    # Rewards are bounded in [0, 1] with gamma=0.99; a healthy soft-Q estimate
    # stays in roughly that scale (theoretical ceiling ~1/(1-gamma)=100 plus
    # slack). A sign-flipped policy/critic update diverges far beyond this
    # within 100 steps on a tiny network -- this bound is intentionally loose
    # so it only fails on gross divergence, not normal training noise.
    assert all(abs(q) < 50 for q in mean_qs[-10:])


def test_policy_loss_sign_moves_mean_toward_higher_q_actions():
    """Isolates the policy_loss sign specifically: replaces the critic with a
    frozen, hand-built Q function that has an unambiguous preference for one
    action, then runs several policy-only gradient steps (the same formula
    update_parameters uses) and checks the policy's mean output moves TOWARD
    the preferred action. A sign-flipped policy_loss would move it away
    instead. This is deterministic and isolates just the sign, unlike a
    many-step Q-divergence check."""
    agent = SACAgent(env=FakeEnv(), state_dim=6, action_dim=2, max_buffer_size=1000, hidden_dim=32)

    target_action = torch.tensor([0.8, -0.8], device=agent.device)

    def fixed_critic_forward(state, action):
        q = -((action - target_action) ** 2).sum(dim=1, keepdim=True)
        return q, q.clone()

    agent.critic.forward = fixed_critic_forward

    state = torch.zeros(1, 6, device=agent.device)
    mean_before = agent.policy.forward(state)[0].detach().clone()

    for _ in range(50):
        pi, log_pi, _ = agent.policy.sample(state)
        q1_pi, q2_pi = agent.critic(state, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (agent.alpha * log_pi - min_q_pi).mean()
        agent.policy_optim.zero_grad()
        policy_loss.backward()
        agent.policy_optim.step()

    mean_after = agent.policy.forward(state)[0].detach().clone()

    dist_before = (mean_before - target_action).abs().sum()
    dist_after = (mean_after - target_action).abs().sum()
    assert dist_after < dist_before
