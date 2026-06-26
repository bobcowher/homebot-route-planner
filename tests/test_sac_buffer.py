import numpy as np
import torch
from sac_buffer import SACReplayBuffer

N_ACTIONS = 8


def _img():
    return torch.randint(0, 256, (3, 96, 96), dtype=torch.uint8)


def _fill(buf, n):
    img = _img()
    goal = np.array([100.0, 200.0], dtype=np.float32)
    motion = np.zeros(4, dtype=np.float32)
    for i in range(n):
        buf.store_transition(img, goal, motion, i % N_ACTIONS, float(np.random.rand()),
                             img, goal, motion, False)


def test_store_and_sample_shapes():
    buf = SACReplayBuffer(max_size=1000, device='cpu')
    _fill(buf, 200)
    batch = buf.sample_buffer(32)
    imgs, goals, motions, actions, rewards, next_imgs, next_goals, next_motions, dones = batch
    assert imgs.shape    == (32, 3, 96, 96)
    assert goals.shape   == (32, 2)
    assert motions.shape == (32, 4)
    assert actions.shape == (32,)
    assert actions.dtype == torch.int64
    assert rewards.shape == (32,)
    assert next_imgs.shape   == (32, 3, 96, 96)
    assert next_goals.shape  == (32, 2)
    assert next_motions.shape == (32, 4)
    assert dones.shape   == (32,)


def test_actions_stored_as_int64():
    buf = SACReplayBuffer(max_size=100, device='cpu')
    img = _img()
    goal = np.zeros(2, dtype=np.float32)
    motion = np.zeros(4, dtype=np.float32)
    buf.store_transition(img, goal, motion, 5, 0.0, img, goal, motion, False)
    assert buf.actions.dtype == torch.int64
    assert buf.actions[0].item() == 5


def test_images_stored_as_uint8_sampled_as_float():
    buf = SACReplayBuffer(max_size=100, device='cpu')
    img = torch.full((3, 96, 96), 128, dtype=torch.uint8)
    goal = np.zeros(2, dtype=np.float32)
    motion = np.zeros(4, dtype=np.float32)
    buf.store_transition(img, goal, motion, 0, 0.0, img, goal, motion, False)
    assert buf.images.dtype == torch.uint8
    imgs, *_ = buf.sample_buffer(1)
    assert imgs.dtype == torch.float32
    assert torch.allclose(imgs, torch.full_like(imgs, 128.0 / 255.0))


def test_buffer_wraps_around():
    buf = SACReplayBuffer(max_size=50, device='cpu')
    _fill(buf, 80)
    assert buf.mem_ctr == 80
    assert min(buf.mem_ctr, buf.mem_size) == 50


def test_can_sample_threshold():
    buf = SACReplayBuffer(max_size=1000, device='cpu')
    batch_size = 16
    _fill(buf, batch_size * 10 - 1)
    assert not buf.can_sample(batch_size)
    _fill(buf, 1)
    assert buf.can_sample(batch_size)
