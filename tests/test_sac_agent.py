import numpy as np
import gymnasium as gym
import torch
from sac_agent import SACAgent


class FakeEnv:
    """Minimal stand-in — only what SACAgent's constructor touches."""
    action_space = gym.spaces.Discrete(8)


def _make_agent(**kwargs):
    return SACAgent(env=FakeEnv(), max_buffer_size=2000, **kwargs)


def _fill_buffer(agent, n):
    img = torch.randint(0, 256, (3, 96, 96), dtype=torch.uint8)
    goal = np.array([100.0, 100.0], dtype=np.float32)
    motion = np.zeros(4, dtype=np.float32)
    for i in range(n):
        agent.memory.store_transition(
            img, goal, motion, i % agent.n_actions, float(np.random.rand()),
            img, goal, motion, False,
        )


def test_select_action_returns_valid_index():
    agent = _make_agent()
    obs_tensor = torch.randint(0, 256, (3, 96, 96), dtype=torch.uint8)
    goal_np    = np.array([100.0, 200.0], dtype=np.float32)
    motion_np  = np.zeros(4, dtype=np.float32)
    action = agent.select_action(obs_tensor, goal_np, motion_np)
    assert isinstance(action, int)
    assert 0 <= action < agent.n_actions


def test_select_action_evaluate_is_deterministic():
    agent = _make_agent()
    obs_tensor = torch.randint(0, 256, (3, 96, 96), dtype=torch.uint8)
    goal_np    = np.array([100.0, 200.0], dtype=np.float32)
    motion_np  = np.zeros(4, dtype=np.float32)
    a1 = agent.select_action(obs_tensor, goal_np, motion_np, evaluate=True)
    a2 = agent.select_action(obs_tensor, goal_np, motion_np, evaluate=True)
    assert a1 == a2


def test_update_parameters_runs_without_nan_and_returns_four_floats():
    agent = _make_agent()
    _fill_buffer(agent, 200)
    critic_loss, actor_loss, mean_q, entropy = agent.update_parameters(batch_size=16)
    assert np.isfinite(critic_loss)
    assert np.isfinite(actor_loss)
    assert np.isfinite(mean_q)
    assert np.isfinite(entropy)
    assert entropy > 0.0  # must be positive (Shannon entropy)


def test_update_parameters_changes_policy_weights():
    agent = _make_agent()
    _fill_buffer(agent, 200)
    before = agent.policy.fc[0].weight.clone().detach()
    agent.update_parameters(batch_size=16)
    after = agent.policy.fc[0].weight.clone().detach()
    assert not torch.allclose(before, after)


def test_polyak_target_update_moves_toward_online_not_away():
    agent = _make_agent(tau=0.1)
    _fill_buffer(agent, 200)

    target_before = agent.critic_target.q1[0].weight.clone().detach()
    agent.update_parameters(batch_size=16)
    online_after  = agent.critic.q1[0].weight.clone().detach()
    target_after  = agent.critic_target.q1[0].weight.clone().detach()

    expected = agent.tau * online_after + (1 - agent.tau) * target_before
    assert torch.allclose(target_after, expected, atol=1e-6)


def test_mean_q_stays_bounded_over_repeated_updates():
    agent = _make_agent()
    for i in range(500):
        img = torch.randint(0, 256, (3, 96, 96), dtype=torch.uint8)
        goal = np.array([100.0, 100.0], dtype=np.float32)
        motion = np.zeros(4, dtype=np.float32)
        r = float(np.random.uniform(0.0, 1.0))
        agent.memory.store_transition(img, goal, motion, i % agent.n_actions, r,
                                      img, goal, motion, False)

    mean_qs = []
    for _ in range(100):
        _, _, mean_q, _ = agent.update_parameters(batch_size=32)
        mean_qs.append(mean_q)

    assert all(abs(q) < 50 for q in mean_qs[-10:])


def test_policy_loss_sign_shifts_prob_toward_high_q_action():
    """Sign isolation: frozen critic prefers action 3.
    After policy gradient steps, π(action=3|s) should increase."""
    agent = _make_agent()
    target_action = 3

    def fake_critic(image, goal, motion):  # noqa: ARG001
        q = torch.zeros(image.shape[0], agent.n_actions, device=agent.device)
        q[:, target_action] = 10.0
        return q, q.clone()

    agent.critic.forward = fake_critic

    img    = torch.randint(0, 256, (1, 3, 96, 96), dtype=torch.uint8).float().to(agent.device) / 255.0
    goal   = torch.zeros(1, 2, device=agent.device)
    motion = torch.zeros(1, 4, device=agent.device)

    with torch.no_grad():
        probs_before, _ = agent.policy(img, goal, motion)

    for _ in range(50):
        probs, log_probs = agent.policy(img, goal, motion)
        with torch.no_grad():
            q1, q2 = agent.critic(img, goal, motion)
        min_q = torch.min(q1, q2)
        actor_loss = (probs * (agent.alpha * log_probs - min_q)).sum(dim=1).mean()
        agent.policy_optim.zero_grad()
        actor_loss.backward()
        agent.policy_optim.step()

    with torch.no_grad():
        probs_after, _ = agent.policy(img, goal, motion)

    assert probs_after[0, target_action].item() > probs_before[0, target_action].item()
