"""Smoke-test the depth-4 head (head_layers=4, all 512) on the coords rep.

    conda run -n sac-homebot python scripts/smoke_depth4.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.q_model import QModel

obs = torch.zeros(2, 3, 96, 96)
goal = torch.tensor([[100.0, 50.0, 700.0, 400.0],
                     [0.0, 0.0, 864.0, 576.0]], dtype=torch.float32)

m = QModel(action_dim=8, goal_dim=4, goal_layers=2, head_layers=4)
out = m(obs, goal)
assert out.shape == (2, 8), f"bad output shape {out.shape}"
assert len(m.head) == 4, f"expected 4 head layers, got {len(m.head)}"
n_params = sum(p.numel() for p in m.parameters())
print(f"depth-4: head_layers={len(m.head)} out={tuple(out.shape)} params={n_params:,} OK")
print("DEPTH4 SMOKE OK")
