import numpy as np
import gymnasium as gym
import torch
from sac_agent import SACAgent


class FakeEnv:
    """Minimal stand-in — only what SACAgent's constructor touches."""
    action_space = gym.spaces.Box(low=np.array([-1., -1.], dtype=np.float32),
                                  high=np.array([1., 1.], dtype=np.float32), dtype=np.float32)


def _make_agent(**kwargs):
    return SACAgent(env=FakeEnv(), action_dim=2, max_buffer_size=2000, **kwargs)


def _fill_buffer(agent, n):
    img = torch.randint(0, 256, (3, 96, 96), dtype=torch.uint8)
    goal = np.array([100.0, 100.0], dtype=np.float32)
    motion = np.zeros(4, dtype=np.float32)
    action = np.zeros(2, dtype=np.float32)
    for _ in range(n):
        agent.memory.store_transition(
            img, goal, motion, action, float(np.random.rand()),
            img, goal, motion, False,
        )


def test_select_action_within_bounds():
    agent = _make_agent()
    obs_tensor = torch.randint(0, 256, (3, 96, 96), dtype=torch.uint8)
    goal_np = np.array([100.0, 200.0], dtype=np.float32)
    motion_np = np.zeros(4, dtype=np.float32)
    action = agent.select_action(obs_tensor, goal_np, motion_np)
    assert action.shape == (2,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_select_action_evaluate_is_deterministic():
    agent = _make_agent()
    obs_tensor = torch.randint(0, 256, (3, 96, 96), dtype=torch.uint8)
    goal_np = np.array([100.0, 200.0], dtype=np.float32)
    motion_np = np.zeros(4, dtype=np.float32)
    a1 = agent.select_action(obs_tensor, goal_np, motion_np, evaluate=True)
    a2 = agent.select_action(obs_tensor, goal_np, motion_np, evaluate=True)
    assert np.allclose(a1, a2)


def test_update_parameters_runs_without_nan_and_returns_three_floats():
    agent = _make_agent()
    _fill_buffer(agent, 200)
    critic_loss, policy_loss, mean_q = agent.update_parameters(batch_size=16)
    assert np.isfinite(critic_loss)
    assert np.isfinite(policy_loss)
    assert np.isfinite(mean_q)


def test_update_parameters_changes_policy_weights():
    agent = _make_agent()
    _fill_buffer(agent, 200)
    before = agent.policy.fc1.weight.clone().detach()
    agent.update_parameters(batch_size=16)
    after = agent.policy.fc1.weight.clone().detach()
    assert not torch.allclose(before, after)


def test_polyak_target_update_moves_toward_online_not_away():
    agent = _make_agent(tau=0.1)
    _fill_buffer(agent, 200)

    target_before = agent.critic_target.q1_fc1.weight.clone().detach()
    agent.update_parameters(batch_size=16)
    online_after = agent.critic.q1_fc1.weight.clone().detach()
    target_after = agent.critic_target.q1_fc1.weight.clone().detach()

    expected = agent.tau * online_after + (1 - agent.tau) * target_before
    assert torch.allclose(target_after, expected, atol=1e-6)


def test_mean_q_stays_bounded_over_repeated_updates():
    agent = _make_agent()
    for _ in range(500):
        img = torch.randint(0, 256, (3, 96, 96), dtype=torch.uint8)
        goal = np.array([100.0, 100.0], dtype=np.float32)
        motion = np.zeros(4, dtype=np.float32)
        action = np.zeros(2, dtype=np.float32)
        r = float(np.random.uniform(0.0, 1.0))
        agent.memory.store_transition(img, goal, motion, action, r, img, goal, motion, False)

    mean_qs = []
    for _ in range(100):
        _, _, mean_q = agent.update_parameters(batch_size=32)
        mean_qs.append(mean_q)

    assert all(abs(q) < 50 for q in mean_qs[-10:])


def test_policy_loss_sign_moves_mean_toward_higher_q_actions():
    """Sign isolation: a frozen hand-built Q with a clear preferred action
    should move the policy mean toward that action after gradient steps."""
    agent = _make_agent()

    target_action = torch.tensor([0.8, -0.8], device=agent.device)

    def fixed_critic_forward(_image, _goal, _motion, action):
        q = -((action - target_action) ** 2).sum(dim=1, keepdim=True)
        return q, q.clone()

    agent.critic.forward = fixed_critic_forward

    img = torch.randint(0, 256, (1, 3, 96, 96), dtype=torch.uint8).float().to(agent.device) / 255.0
    goal = torch.zeros(1, 2, device=agent.device)
    motion = torch.zeros(1, 4, device=agent.device)

    mean_before = agent.policy.forward(img, goal, motion)[0].detach().clone()

    for _ in range(50):
        pi, log_pi, _ = agent.policy.sample(img, goal, motion)
        q1_pi, q2_pi = agent.critic(img, goal, motion, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (agent.alpha * log_pi - min_q_pi).mean()
        agent.policy_optim.zero_grad()
        policy_loss.backward()
        agent.policy_optim.step()

    mean_after = agent.policy.forward(img, goal, motion)[0].detach().clone()

    dist_before = (mean_before - target_action).abs().sum()
    dist_after = (mean_after - target_action).abs().sum()
    assert dist_after < dist_before
