"""The shared softmax_rel policy (policy.softmax_rel_probs) used by both the
training rollout (agent, softmax_behavior) and deploy/eval. The contract that
matters: a valid distribution, scale-invariance (a fixed temp transfers across
checkpoints with different Q magnitudes), and that it still prefers the best
action (it's exploit-with-noise, not uniform exploration)."""
import torch

from policy import softmax_rel_probs, decode_macro


def test_decode_macro_round_trip():
    """A macro index decodes to base-n_base digits (MSB first) and re-encodes back."""
    n_base, H = 8, 3
    for idx in (0, 1, 7, 8, 73, 511):
        acts = decode_macro(idx, H, n_base)
        assert len(acts) == H and all(0 <= a < n_base for a in acts)
        re = 0
        for a in acts:
            re = re * n_base + a
        assert re == idx


def test_decode_macro_h1_is_identity():
    """macro_h=1 reduces to [idx] -- the per-step policy is the H=1 special case."""
    for idx in range(8):
        assert decode_macro(idx, 1, 8) == [idx]


def test_returns_valid_distribution():
    q = torch.tensor([1.0, -2.0, 0.5, 3.0, -1.0, 0.0, 2.0, -0.5])
    p = softmax_rel_probs(q, 0.1)
    assert torch.all(p >= 0)
    assert abs(float(p.sum()) - 1.0) < 1e-5


def test_prefers_the_argmax_action():
    q = torch.tensor([0.0, 0.0, 5.0, 0.0])
    p = softmax_rel_probs(q, 0.1)
    assert int(p.argmax()) == int(q.argmax()) == 2


def test_affine_invariance():
    # scale = temp*std and softmax is shift-invariant, so a*q+b (a>0) -> same probs.
    # This is WHY a single temp=0.1 transfers across checkpoints with different Q scales.
    q = torch.tensor([1.0, -2.0, 0.5, 3.0, -1.0, 0.0, 2.0, -0.5])
    base = softmax_rel_probs(q, 0.1)
    scaled = softmax_rel_probs(1000.0 * q + 7.0, 0.1)
    assert torch.allclose(base, scaled, atol=1e-4)


def test_lower_temp_is_sharper():
    q = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    sharp = softmax_rel_probs(q, 0.05)
    soft = softmax_rel_probs(q, 0.5)
    assert float(sharp.max()) > float(soft.max())


def test_exact_tie_is_uniform():
    q = torch.zeros(8)
    p = softmax_rel_probs(q, 0.1)
    assert torch.allclose(p, torch.full((8,), 1.0 / 8), atol=1e-5)
