"""Flat replay buffer for continuous SAC. Same shape as sac-fetch's
buffer.py, trimmed of the Kitchen-specific expert-data/augmentation/CSV
extras this project doesn't use."""
import numpy as np


class SACReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.mem_size = max_size
        self.mem_ctr = 0

        self.state_memory = np.zeros((max_size, state_dim), dtype=np.float32)
        self.next_state_memory = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action_memory = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward_memory = np.zeros(max_size, dtype=np.float32)
        self.terminal_memory = np.zeros(max_size, dtype=bool)

    def can_sample(self, batch_size: int) -> bool:
        return self.mem_ctr >= batch_size * 10

    def store_transition(self, state, action, reward, next_state, done):
        idx = self.mem_ctr % self.mem_size
        self.state_memory[idx] = state
        self.next_state_memory[idx] = next_state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.terminal_memory[idx] = done
        self.mem_ctr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_ctr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        return (self.state_memory[batch], self.action_memory[batch],
                self.reward_memory[batch], self.next_state_memory[batch],
                self.terminal_memory[batch])
