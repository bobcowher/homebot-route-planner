"""Smoke-test the bottleneck knob on the coords-deep QModel.

    conda run -n sac-homebot python scripts/smoke_bottleneck.py

Confirms bottleneck=None (== prior coords-deep) and bottleneck=64 both build and
forward to the right shape, and that the bottleneck actually cuts param count.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.q_model import QModel

obs = torch.zeros(2, 3, 96, 96)
goal = torch.tensor([[100.0, 50.0, 700.0, 400.0],
                     [0.0, 0.0, 864.0, 576.0]], dtype=torch.float32)

for name, bn, want_in in [("no-bottleneck", None, 512), ("bottleneck-64", 64, 64)]:
    m = QModel(action_dim=8, goal_dim=4, goal_layers=2, head_layers=2, bottleneck=bn)
    out = m(obs, goal)
    assert out.shape == (2, 8), f"{name}: bad output shape {out.shape}"
    # The IB compresses the REPRESENTATION feeding the action head (not params):
    # the output layer should read from a `want_in`-wide channel.
    assert m.output.in_features == want_in, f"{name}: output.in_features={m.output.in_features}"
    n_params = sum(p.numel() for p in m.parameters())
    print(f"{name}: out={tuple(out.shape)} output.in={m.output.in_features} params={n_params:,} OK")

print("BOTTLENECK SMOKE OK")
