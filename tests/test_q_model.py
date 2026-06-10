# tests/test_q_model.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
from models.q_model import QModel


def test_forward_returns_correct_shape():
    model = QModel(action_dim=8, input_shape=(3, 96, 96), goal_dim=2)
    obs  = torch.rand(4, 3, 96, 96)
    goal = torch.rand(4, 2)
    q    = model(obs, goal)
    assert q.shape == (4, 8), f"expected (4,8), got {q.shape}"


def test_forward_single_sample():
    model = QModel(action_dim=8, input_shape=(3, 96, 96), goal_dim=2)
    obs  = torch.rand(1, 3, 96, 96)
    goal = torch.rand(1, 2)
    q    = model(obs, goal)
    assert q.shape == (1, 8)
