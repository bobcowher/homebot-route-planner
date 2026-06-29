"""Image-aware replay buffer for discrete SAC+HER.

Actions are stored as int64 indices (not float vectors).
Images stored as uint8, normalised to [0,1] at sample time.
Goals and motions stored as float32.

store_transition signature:
    image      : (3, 96, 96) uint8 tensor
    goal       : (2,)        float32  — noisy_world_vector [dx, dy]
    motion     : (4,)        float32  — [dx/step, dy/step, 0, 0]
    action     : int         — discrete action index
    reward     : float       — n-step return Σ γ^k r_{t+k} (1-step reward if n_step=1)
    next_image : (3, 96, 96) uint8   — bootstrap state s_{t+m}
    next_goal  : (2,)        float32
    next_motion: (4,)        float32
    done       : bool
    discount   : float       — γ^m bootstrap multiplier (m = n-step horizon, truncated at
                               a terminal/episode end). Default 1.0 for direct/legacy stores.

sample_buffer returns:
    (imgs, goals, motions, actions, rewards, next_imgs, next_goals, next_motions, dones, discounts)
    actions: LongTensor (B,) — ready for .gather(1, actions.unsqueeze(1))
"""
import os

import torch


class SACReplayBuffer:
    IMAGE_SHAPE = (3, 96, 96)
    GOAL_DIM    = 2
    MOTION_DIM  = 4

    def __init__(self, max_size, device='cpu'):
        self.mem_size = max_size
        self.mem_ctr  = 0
        self.device   = device

        override = os.getenv("REPLAY_BUFFER_MEMORY")
        if override in ("cpu", "cuda:0", "cuda:1"):
            self.device = override

        self.images       = torch.zeros((max_size, *self.IMAGE_SHAPE), dtype=torch.uint8,   device=self.device)
        self.next_images  = torch.zeros((max_size, *self.IMAGE_SHAPE), dtype=torch.uint8,   device=self.device)
        self.goals        = torch.zeros((max_size, self.GOAL_DIM),     dtype=torch.float32, device=self.device)
        self.next_goals   = torch.zeros((max_size, self.GOAL_DIM),     dtype=torch.float32, device=self.device)
        self.motions      = torch.zeros((max_size, self.MOTION_DIM),   dtype=torch.float32, device=self.device)
        self.next_motions = torch.zeros((max_size, self.MOTION_DIM),   dtype=torch.float32, device=self.device)
        self.actions      = torch.zeros(max_size,                      dtype=torch.int64,   device=self.device)
        self.rewards      = torch.zeros(max_size,                      dtype=torch.float32, device=self.device)
        self.dones        = torch.zeros(max_size,                      dtype=torch.bool,    device=self.device)
        self.discounts    = torch.ones(max_size,                       dtype=torch.float32, device=self.device)

    def can_sample(self, batch_size: int) -> bool:
        return self.mem_ctr >= batch_size * 10

    def store_transition(self, image, goal, motion, action, reward,
                         next_image, next_goal, next_motion, done, discount=1.0):
        idx = self.mem_ctr % self.mem_size
        self.images[idx]       = torch.as_tensor(image,      dtype=torch.uint8,   device=self.device)
        self.next_images[idx]  = torch.as_tensor(next_image, dtype=torch.uint8,   device=self.device)
        self.goals[idx]        = torch.as_tensor(goal,       dtype=torch.float32, device=self.device)
        self.next_goals[idx]   = torch.as_tensor(next_goal,  dtype=torch.float32, device=self.device)
        self.motions[idx]      = torch.as_tensor(motion,     dtype=torch.float32, device=self.device)
        self.next_motions[idx] = torch.as_tensor(next_motion,dtype=torch.float32, device=self.device)
        self.actions[idx]      = int(action)
        self.rewards[idx]      = float(reward)
        self.dones[idx]        = bool(done)
        self.discounts[idx]    = float(discount)
        self.mem_ctr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_ctr, self.mem_size)
        idx = torch.randint(0, max_mem, (batch_size,), device=self.device)
        return (
            self.images[idx].float() / 255.0,
            self.goals[idx],
            self.motions[idx],
            self.actions[idx],           # LongTensor (B,)
            self.rewards[idx],
            self.next_images[idx].float() / 255.0,
            self.next_goals[idx],
            self.next_motions[idx],
            self.dones[idx],
            self.discounts[idx],
        )
