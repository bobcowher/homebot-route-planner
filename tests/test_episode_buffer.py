# tests/test_episode_buffer.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
import numpy as np
from episode_buffer import EpisodeBuffer
from buffer import ReplayBuffer


def _make_replay():
    return ReplayBuffer(
        max_size=10000,
        input_shape=(3, 96, 96),
        input_device='cpu',
        output_device='cpu',
    )


def _dummy_compute_reward(ag, dg, info):
    return np.zeros(len(ag), dtype=np.float32)


def test_send_to_transition_count():
    """10-step episode with K=4 future strategy.

    Original: 10 transitions.
    Hindsight per step: min(4, len(future)) — step 9 has no future, skipped.
      steps 0-5: 4 each  =  24
      step 6: 3, step 7: 2, step 8: 1, step 9: 0  = 6
    Total: 10 + 30 = 40
    """
    ep  = EpisodeBuffer()
    rep = _make_replay()
    obs = torch.zeros(3, 96, 96, dtype=torch.uint8)
    desired_goal = np.array([300.0, 400.0], dtype=np.float32)

    for i in range(10):
        ep.store(obs, 0, 0.0, obs, False,
                 achieved_goal=np.array([float(i * 10), float(i * 10)], dtype=np.float32))

    ep.send_to(rep, desired_goal, _dummy_compute_reward)
    assert rep.mem_ctr == 40, f"expected 40, got {rep.mem_ctr}"


def test_send_to_clears_nothing_on_its_own():
    ep  = EpisodeBuffer()
    rep = _make_replay()
    obs = torch.zeros(3, 96, 96, dtype=torch.uint8)
    ag  = np.array([0.0, 0.0], dtype=np.float32)
    dg  = np.array([100.0, 100.0], dtype=np.float32)

    ep.store(obs, 0, 0.0, obs, False, achieved_goal=ag)
    ep.send_to(rep, dg, _dummy_compute_reward)
    assert len(ep) == 1, "send_to must not clear the buffer — caller does that"


def test_send_to_original_reward_preserved():
    ep  = EpisodeBuffer()
    rep = _make_replay()
    obs = torch.zeros(3, 96, 96, dtype=torch.uint8)
    dg  = np.array([100.0, 100.0], dtype=np.float32)
    ag  = np.array([0.0, 0.0], dtype=np.float32)

    ep.store(obs, 0, 7.0, obs, False, achieved_goal=ag)
    ep.send_to(rep, dg, _dummy_compute_reward)

    # First stored transition is the original — reward must be 7.0
    assert float(rep.reward_memory[0]) == 7.0
