import torch
from model import DiscreteQNet, DiscretePolicy

N_ACTIONS = 8
BATCH     = 4


def _inputs(batch=BATCH):
    image  = torch.rand(batch, 3, 96, 96)
    goal   = torch.rand(batch, 2)
    motion = torch.rand(batch, 4)
    return image, goal, motion


def test_qnet_forward_shapes():
    qnet = DiscreteQNet(n_actions=N_ACTIONS)
    image, goal, motion = _inputs()
    q1, q2 = qnet(image, goal, motion)
    assert q1.shape == (BATCH, N_ACTIONS)
    assert q2.shape == (BATCH, N_ACTIONS)


def test_policy_forward_shapes_and_valid_probs():
    policy = DiscretePolicy(n_actions=N_ACTIONS)
    image, goal, motion = _inputs()
    probs, log_probs = policy(image, goal, motion)
    assert probs.shape == (BATCH, N_ACTIONS)
    assert log_probs.shape == (BATCH, N_ACTIONS)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(BATCH), atol=1e-5)


def test_policy_get_action_returns_valid_index():
    policy = DiscretePolicy(n_actions=N_ACTIONS)
    image, goal, motion = _inputs(batch=2)
    action, log_probs = policy.get_action(image, goal, motion)
    assert action.shape == (2,)
    assert log_probs.shape == (2, N_ACTIONS)
    assert torch.all(action >= 0) and torch.all(action < N_ACTIONS)


def test_policy_evaluate_is_deterministic():
    policy = DiscretePolicy(n_actions=N_ACTIONS)
    image, goal, motion = _inputs(batch=1)
    a1, _ = policy.get_action(image, goal, motion, evaluate=True)
    a2, _ = policy.get_action(image, goal, motion, evaluate=True)
    assert a1.item() == a2.item()
