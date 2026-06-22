"""Shared action-selection policy used by BOTH training rollout (agent.select_action
with softmax_behavior) and deploy/eval (chained_eval, spin_metric, navigator_tool).

Keeping a single definition is the point: the whole softmax-behavior experiment is
"train under the policy we deploy", which is only true if both call the same code.
"""
import torch
import torch.nn.functional as F


def decode_macro(idx: int, macro_h: int, n_base: int) -> list[int]:
    """Macro index -> list of `macro_h` base actions (base-`n_base` digits, most-
    significant first). The inverse of sum(a_i * n_base**(H-1-i)). Shared by the
    training rollout (agent) and every eval path so train==deploy decoding is one
    definition. macro_h=1 returns [idx] -- identical to the per-step policy."""
    acts = []
    for _ in range(macro_h):
        acts.append(idx % n_base)
        idx //= n_base
    return acts[::-1]


def softmax_rel_probs(q: torch.Tensor, temp: float) -> torch.Tensor:
    """Scale-invariant ("relative") softmax over a 1-D Q vector.

    Temperature is a unitless fraction of the per-state Q spread, so it transfers
    across checkpoints regardless of Q magnitude: scale = temp * std(Q). A near-tie
    (tiny std) -> ~uniform (explore the tie); a confident state (large std) -> sharp
    (exploit). Invariant to affine rescaling of Q (a*Q + b, a>0) by construction,
    which is why a fixed temp works across checkpoints with different Q scales.
    """
    scale = temp * (q.std() + 1e-8)
    return F.softmax(q / scale, dim=-1)
