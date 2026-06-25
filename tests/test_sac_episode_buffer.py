import numpy as np
import torch
from sac_episode_buffer import SACEpisodeBuffer


class FakeReplayBuffer:
    def __init__(self):
        self.calls = []

    def store_transition(self, image, goal, motion, action, reward,
                         next_image, next_goal, next_motion, done):
        self.calls.append((image, goal, motion, action, reward,
                           next_image, next_goal, next_motion, done))


def fake_compute_reward(achieved, desired, info):
    diff = np.asarray(achieved, dtype=np.float32) - np.asarray(desired, dtype=np.float32)
    dist = np.linalg.norm(diff, axis=-1)
    return (dist <= 31.0).astype(np.float32)


def _obs():
    return torch.zeros(3, 96, 96, dtype=torch.uint8)


def _motion():
    return np.zeros(4, dtype=np.float32)


def test_empty_buffer_sends_nothing():
    eb = SACEpisodeBuffer()
    rb = FakeReplayBuffer()
    eb.send_to(rb, desired_goal=np.array([0.0, 0.0]),
               compute_reward=fake_compute_reward, goal_noise_std=0.0)
    assert rb.calls == []


def test_single_transition_real_goal_is_world_displacement():
    eb = SACEpisodeBuffer()
    obs = _obs()
    eb.store(
        obs=obs, next_obs=obs, action=np.array([1.0, 0.0], dtype=np.float32),
        reward=0.0, done=False,
        achieved_prev=np.array([100.0, 100.0]),
        achieved_next=np.array([104.0, 100.0]),
        motion_prev=_motion(), motion_next=_motion(),
    )
    rb = FakeReplayBuffer()
    desired_goal = np.array([200.0, 100.0])
    eb.send_to(rb, desired_goal=desired_goal, compute_reward=fake_compute_reward,
               goal_noise_std=0.0, k=0)

    assert len(rb.calls) == 1
    _, goal, _, _, _, _, next_goal, _, _ = rb.calls[0]
    # goal = noisy_world_vector(100,100, 200,100, noise=0) = [100, 0]
    assert np.allclose(goal, [200.0 - 100.0, 100.0 - 100.0])
    # next_goal = noisy_world_vector(104,100, 200,100, noise=0) = [96, 0]
    assert np.allclose(next_goal, [200.0 - 104.0, 100.0 - 100.0])


def test_hindsight_relabel_produces_terminal_success():
    eb = SACEpisodeBuffer()
    obs = _obs()
    # Step 0: robot (0,0) → (10,0)
    eb.store(obs=obs, next_obs=obs, action=np.array([1.0, 0.0], dtype=np.float32),
             reward=0.0, done=False,
             achieved_prev=np.array([0.0, 0.0]),
             achieved_next=np.array([10.0, 0.0]),
             motion_prev=_motion(), motion_next=_motion())
    # Step 1: robot (10,0) → (20,0)
    eb.store(obs=obs, next_obs=obs, action=np.array([1.0, 0.0], dtype=np.float32),
             reward=0.0, done=False,
             achieved_prev=np.array([10.0, 0.0]),
             achieved_next=np.array([20.0, 0.0]),
             motion_prev=_motion(), motion_next=_motion())

    rb = FakeReplayBuffer()
    # k=1: step 0 gets one hindsight goal = t1.achieved_next = (20,0)
    # dist([10,0], [20,0]) = 10 < 31 → reward=1.0, done=True
    eb.send_to(rb, desired_goal=np.array([999.0, 999.0]),
               compute_reward=fake_compute_reward, goal_noise_std=0.0, k=1)

    assert len(rb.calls) == 3  # 2 real + 1 hindsight
    _, _, _, _, reward, _, _, _, done = rb.calls[-1]
    assert reward == 1.0
    assert done is True


def test_clear_empties_buffer():
    eb = SACEpisodeBuffer()
    obs = _obs()
    eb.store(obs=obs, next_obs=obs, action=np.zeros(2, dtype=np.float32),
             reward=0.0, done=False,
             achieved_prev=np.zeros(2), achieved_next=np.zeros(2),
             motion_prev=_motion(), motion_next=_motion())
    assert len(eb) == 1
    eb.clear()
    assert len(eb) == 0
