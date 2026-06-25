"""Image-aware replay buffer for CNN-based SAC+HER.

Mirrors the DQN champion's buffer.py: images stored as uint8 tensors on
device (cheap storage, normalised to [0,1] at sample time), goals and motions
stored as float32 tensors. Separate goal/motion arrays let HER swap goals
without re-running the CNN.

store_transition signature:
    image      : (3, 96, 96) uint8 tensor or ndarray
    goal       : (2,)        float32  — noisy_world_vector [dx, dy]
    motion     : (4,)        float32  — [last_linear, last_angular, dx, dy]
    action     : (action_dim,) float32
    reward     : float
    next_image : (3, 96, 96) uint8
    next_goal  : (2,)        float32
    next_motion: (4,)        float32
    done       : bool
"""
import os

import torch


class SACReplayBuffer:
    IMAGE_SHAPE = (3, 96, 96)
    GOAL_DIM = 2
    MOTION_DIM = 4

    def __init__(self, max_size, action_dim, device='cpu'):
        self.mem_size = max_size
        self.mem_ctr = 0
        self.device = device

        override = os.getenv("REPLAY_BUFFER_MEMORY")
        if override in ("cpu", "cuda:0", "cuda:1"):
            self.device = override

        self.images       = torch.zeros((max_size, *self.IMAGE_SHAPE), dtype=torch.uint8,   device=self.device)
        self.next_images  = torch.zeros((max_size, *self.IMAGE_SHAPE), dtype=torch.uint8,   device=self.device)
        self.goals        = torch.zeros((max_size, self.GOAL_DIM),     dtype=torch.float32, device=self.device)
        self.next_goals   = torch.zeros((max_size, self.GOAL_DIM),     dtype=torch.float32, device=self.device)
        self.motions      = torch.zeros((max_size, self.MOTION_DIM),   dtype=torch.float32, device=self.device)
        self.next_motions = torch.zeros((max_size, self.MOTION_DIM),   dtype=torch.float32, device=self.device)
        self.actions      = torch.zeros((max_size, action_dim),        dtype=torch.float32, device=self.device)
        self.rewards      = torch.zeros(max_size,                      dtype=torch.float32, device=self.device)
        self.dones        = torch.zeros(max_size,                      dtype=torch.bool,    device=self.device)

    def can_sample(self, batch_size: int) -> bool:
        return self.mem_ctr >= batch_size * 10

    def store_transition(self, image, goal, motion, action, reward,
                         next_image, next_goal, next_motion, done):
        idx = self.mem_ctr % self.mem_size
        self.images[idx]       = torch.as_tensor(image,       dtype=torch.uint8,   device=self.device)
        self.next_images[idx]  = torch.as_tensor(next_image,  dtype=torch.uint8,   device=self.device)
        self.goals[idx]        = torch.as_tensor(goal,        dtype=torch.float32, device=self.device)
        self.next_goals[idx]   = torch.as_tensor(next_goal,   dtype=torch.float32, device=self.device)
        self.motions[idx]      = torch.as_tensor(motion,      dtype=torch.float32, device=self.device)
        self.next_motions[idx] = torch.as_tensor(next_motion, dtype=torch.float32, device=self.device)
        self.actions[idx]      = torch.as_tensor(action,      dtype=torch.float32, device=self.device)
        self.rewards[idx]      = float(reward)
        self.dones[idx]        = bool(done)
        self.mem_ctr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_ctr, self.mem_size)
        idx = torch.randint(0, max_mem, (batch_size,), device=self.device)
        return (
            self.images[idx].float() / 255.0,
            self.goals[idx],
            self.motions[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_images[idx].float() / 255.0,
            self.next_goals[idx],
            self.next_motions[idx],
            self.dones[idx],
        )
