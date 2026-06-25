import numpy as np
from sac_buffer import SACReplayBuffer


def test_can_sample_false_when_empty():
    buf = SACReplayBuffer(max_size=1000, state_dim=6, action_dim=2)
    assert buf.can_sample(batch_size=8) is False


def test_can_sample_true_after_enough_transitions():
    buf = SACReplayBuffer(max_size=1000, state_dim=6, action_dim=2)
    for _ in range(80):
        buf.store_transition(np.zeros(6), np.zeros(2), 0.0, np.zeros(6), False)
    assert buf.can_sample(batch_size=8) is True


def test_store_and_sample_roundtrip():
    buf = SACReplayBuffer(max_size=1000, state_dim=6, action_dim=2)
    state = np.arange(6, dtype=np.float32)
    action = np.array([0.5, -0.5], dtype=np.float32)
    buf.store_transition(state, action, 1.0, state * 2, True)

    states, actions, rewards, next_states, dones = buf.sample_buffer(batch_size=1)
    assert np.allclose(states[0], state)
    assert np.allclose(actions[0], action)
    assert rewards[0] == 1.0
    assert np.allclose(next_states[0], state * 2)
    assert dones[0] == True


def test_wraps_around_at_capacity():
    buf = SACReplayBuffer(max_size=4, state_dim=2, action_dim=1)
    for i in range(7):
        buf.store_transition(np.array([i, i], dtype=np.float32), np.array([i], dtype=np.float32),
                             float(i), np.array([i, i], dtype=np.float32), False)
    assert buf.mem_ctr == 7
    # index 6 % 4 == 2 was the last write -> slot 2 holds transition i=2 overwritten by i=6
    assert buf.state_memory[2][0] == 6.0
