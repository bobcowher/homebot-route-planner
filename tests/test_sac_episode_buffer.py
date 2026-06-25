import numpy as np
from sac_episode_buffer import SACEpisodeBuffer
from goal_geometry import ego_vector


class FakeReplayBuffer:
    """Captures store_transition calls for assertion instead of a real buffer."""
    def __init__(self):
        self.calls = []

    def store_transition(self, state, action, reward, next_state, done):
        self.calls.append((state, action, reward, next_state, done))


def fake_compute_reward(achieved, desired, info):
    diff = np.asarray(achieved, dtype=np.float32) - np.asarray(desired, dtype=np.float32)
    dist = np.linalg.norm(diff, axis=-1)
    return (dist <= 31.0).astype(np.float32)


def test_empty_buffer_sends_nothing():
    eb = SACEpisodeBuffer()
    rb = FakeReplayBuffer()
    eb.send_to(rb, desired_goal=np.array([0.0, 0.0]), compute_reward=fake_compute_reward)
    assert rb.calls == []


def test_single_transition_real_goal_state_matches_ego_vector():
    eb = SACEpisodeBuffer()
    motion = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    eb.store(
        action=np.array([1.0, 0.0]), reward=0.0, done=False,
        achieved_prev=np.array([100.0, 100.0]), achieved_next=np.array([104.0, 100.0]),
        heading_prev=0.0, heading_next=0.0,
        motion_prev=motion, motion_next=motion,
    )
    rb = FakeReplayBuffer()
    desired_goal = np.array([200.0, 100.0])
    eb.send_to(rb, desired_goal=desired_goal, compute_reward=fake_compute_reward, k=0)

    assert len(rb.calls) == 1
    state, action, reward, next_state, done = rb.calls[0]
    expected_goal = ego_vector(100.0, 100.0, 0.0, 200.0, 100.0)
    assert np.allclose(state[:2], expected_goal)
    assert np.allclose(state[2:], motion)
    assert reward == 0.0
    assert done is False


def test_hindsight_relabel_produces_terminal_success():
    eb = SACEpisodeBuffer()
    motion = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    # Step 0: far from any goal.
    eb.store(action=np.array([1.0, 0.0]), reward=0.0, done=False,
             achieved_prev=np.array([0.0, 0.0]), achieved_next=np.array([10.0, 0.0]),
             heading_prev=0.0, heading_next=0.0, motion_prev=motion, motion_next=motion)
    # Step 1: lands exactly on what step 0 will be relabeled toward.
    eb.store(action=np.array([1.0, 0.0]), reward=0.0, done=False,
             achieved_prev=np.array([10.0, 0.0]), achieved_next=np.array([20.0, 0.0]),
             heading_prev=0.0, heading_next=0.0, motion_prev=motion, motion_next=motion)

    rb = FakeReplayBuffer()
    # k=1 forces exactly one hindsight relabel per eligible transition.
    eb.send_to(rb, desired_goal=np.array([999.0, 999.0]), compute_reward=fake_compute_reward, k=1)

    # 2 original + 1 hindsight (only step 0 has a future transition to sample from).
    assert len(rb.calls) == 3
    hindsight_call = rb.calls[-1]
    _, _, reward, _, done = hindsight_call
    assert reward == 1.0  # achieved_next (10,0) == hindsight goal (20,0)'s achieved_next? see below
    assert done is True


def test_clear_empties_buffer():
    eb = SACEpisodeBuffer()
    eb.store(action=np.array([0.0, 0.0]), reward=0.0, done=False,
             achieved_prev=np.array([0.0, 0.0]), achieved_next=np.array([0.0, 0.0]),
             heading_prev=0.0, heading_next=0.0,
             motion_prev=np.zeros(4), motion_next=np.zeros(4))
    assert len(eb) == 1
    eb.clear()
    assert len(eb) == 0
