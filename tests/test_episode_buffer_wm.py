# tests/test_episode_buffer_wm.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
from buffer import EpisodeReplayBuffer
from models.detection_head import K_LABEL_SLOTS


def _buf():
    return EpisodeReplayBuffer(max_size=200, input_shape=(3, 96, 96),
                               input_device="cpu", output_device="cpu", action_dim=2)


def _labels(x):
    rows = torch.full((K_LABEL_SLOTS, 3), -1, dtype=torch.int16)
    rows[0] = torch.tensor([0, x, x], dtype=torch.int16)
    return rows


def test_sequences_return_labels_aligned():
    buf = _buf()
    obs = torch.zeros(3, 96, 96, dtype=torch.uint8)
    # one episode of 10 steps; label x == step index
    for t in range(10):
        done = (t == 9)
        buf.store_transition(obs, [0.0, 0.0], 0.0, obs, done, done, _labels(t))
    batch = buf.sample_sequences(batch_size=5, sequence_length=5)
    assert batch["labels"].shape == (1, 5, K_LABEL_SLOTS, 3)
    # within the sampled contiguous window, label x increments by 1 each step
    xs = batch["labels"][0, :, 0, 1]
    assert torch.all(xs[1:] - xs[:-1] == 1)


def test_can_sample_sequences_gate():
    buf = _buf()
    obs = torch.zeros(3, 96, 96, dtype=torch.uint8)
    for t in range(5):
        buf.store_transition(obs, [0.0, 0.0], 0.0, obs, t == 4, t == 4, _labels(t))
    # only one short episode -> cannot sample 10-long sequences
    assert not buf.can_sample_sequences(batch_size=10, sequence_length=10)
