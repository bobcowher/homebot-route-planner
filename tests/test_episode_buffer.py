import numpy as np
import torch
from episode_buffer import SACEpisodeBuffer


class FakeReplayBuffer:
    def __init__(self):
        self.calls = []

    def store_transition(self, image, goal, motion, action, reward,
                         next_image, next_goal, next_motion, done, discount=1.0):
        self.calls.append((image, goal, motion, action, reward,
                           next_image, next_goal, next_motion, done, discount))


def fake_compute_reward(achieved, desired, info):  # noqa: ARG001
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


def test_single_transition_action_stored_as_int():
    eb = SACEpisodeBuffer()
    obs = _obs()
    eb.store(obs=obs, next_obs=obs, action=3, reward=0.0, done=False,
             achieved_prev=np.zeros(2), achieved_next=np.zeros(2),
             motion_prev=_motion(), motion_next=_motion())
    assert eb._transitions[0].action == 3
    assert isinstance(eb._transitions[0].action, int)


def test_single_transition_real_goal_is_world_displacement():
    eb = SACEpisodeBuffer()
    obs = _obs()
    eb.store(obs=obs, next_obs=obs, action=1, reward=0.0, done=False,
             achieved_prev=np.array([100.0, 100.0]),
             achieved_next=np.array([104.0, 100.0]),
             motion_prev=_motion(), motion_next=_motion())
    rb = FakeReplayBuffer()
    desired_goal = np.array([200.0, 100.0])
    eb.send_to(rb, desired_goal=desired_goal,
               compute_reward=fake_compute_reward, goal_noise_std=0.0, k=0)

    assert len(rb.calls) == 1
    _, goal, _, action, _, _, next_goal, _, _, _ = rb.calls[0]
    assert np.allclose(goal,      [200.0 - 100.0, 100.0 - 100.0])
    assert np.allclose(next_goal, [200.0 - 104.0, 100.0 - 100.0])
    assert action == 1


def test_hindsight_relabel_produces_terminal_success():
    eb = SACEpisodeBuffer()
    obs = _obs()
    eb.store(obs=obs, next_obs=obs, action=0, reward=0.0, done=False,
             achieved_prev=np.array([0.0, 0.0]),
             achieved_next=np.array([10.0, 0.0]),
             motion_prev=_motion(), motion_next=_motion())
    eb.store(obs=obs, next_obs=obs, action=0, reward=0.0, done=False,
             achieved_prev=np.array([10.0, 0.0]),
             achieved_next=np.array([20.0, 0.0]),
             motion_prev=_motion(), motion_next=_motion())

    rb = FakeReplayBuffer()
    eb.send_to(rb, desired_goal=np.array([999.0, 999.0]),
               compute_reward=fake_compute_reward, goal_noise_std=0.0, k=1)

    assert len(rb.calls) == 3
    _, _, _, _, reward, _, _, _, done, _ = rb.calls[-1]
    assert reward == 1.0
    assert done is True


def test_nstep_return_discounts_terminal_reward_and_sets_discount():
    """3-step return: goal reached on the 3rd step of the window -> stored reward = γ^2·1,
    done=True, bootstrap discount = γ^3 (masked out by done). Earlier transitions whose
    window does NOT reach the goal store R=0 and the full γ^3 bootstrap discount."""
    eb = SACEpisodeBuffer(n_step=3, gamma=0.99)
    obs = _obs()
    # achieved_next distances to goal [100,0]: 90, 60, 20(reach), 5, ...
    for ax in (10.0, 40.0, 80.0, 95.0, 98.0):
        eb.store(obs=obs, next_obs=obs, action=0, reward=0.0, done=False,
                 achieved_prev=np.array([ax - 5.0, 0.0]),
                 achieved_next=np.array([ax, 0.0]),
                 motion_prev=_motion(), motion_next=_motion())

    rb = FakeReplayBuffer()
    eb.send_to(rb, desired_goal=np.array([100.0, 0.0]),
               compute_reward=fake_compute_reward, goal_noise_std=0.0, k=0)

    # call 0 = real-goal n-step from transition 0: reaches goal at window step 3 (m=2)
    _, _, _, _, reward0, _, _, _, done0 = rb.calls[0][:9]
    assert abs(reward0 - 0.99 ** 2) < 1e-6
    assert done0 is True
    assert abs(rb.calls[0][9] - 0.99 ** 3) < 1e-6  # discount γ^m, m=3 (masked by done)


def test_nstep_nonterminal_window_bootstraps_full_discount():
    """A window that never reaches the goal stores R=0, done=False, discount γ^n_step."""
    eb = SACEpisodeBuffer(n_step=3, gamma=0.99)
    obs = _obs()
    for _ in range(5):
        eb.store(obs=obs, next_obs=obs, action=0, reward=0.0, done=False,
                 achieved_prev=np.array([0.0, 0.0]),
                 achieved_next=np.array([0.0, 0.0]),
                 motion_prev=_motion(), motion_next=_motion())

    rb = FakeReplayBuffer()
    eb.send_to(rb, desired_goal=np.array([999.0, 999.0]),
               compute_reward=fake_compute_reward, goal_noise_std=0.0, k=0)

    _, _, _, _, reward0, _, _, _, done0 = rb.calls[0][:9]
    assert reward0 == 0.0
    assert done0 is False
    assert abs(rb.calls[0][9] - 0.99 ** 3) < 1e-6


def test_clear_empties_buffer():
    eb = SACEpisodeBuffer()
    obs = _obs()
    eb.store(obs=obs, next_obs=obs, action=0, reward=0.0, done=False,
             achieved_prev=np.zeros(2), achieved_next=np.zeros(2),
             motion_prev=_motion(), motion_next=_motion())
    assert len(eb) == 1
    eb.clear()
    assert len(eb) == 0
