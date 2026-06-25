"""CNN-based double-Q SAC actor/critic for HomeBotGoalEnv.

Architecture mirrors the DQN champion (QModel): shared CNN structure across
Policy and Critic (separate weights), goal encoder (2-layer, 128 hidden),
motion encoder (linear, 32), combined head feeding the actor/critic outputs.

Inputs per forward call:
  image : (B, 3, 96, 96) float32 in [0, 1]  — normalized from uint8
  goal  : (B, 2)          float32             — noisy_world_vector [dx, dy]
  motion: (B, 4)          float32             — [last_linear, last_angular, dx, dy]

For Critic, action (B, action_dim) is additionally concatenated into the head.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -4
epsilon = 1e-6

GOAL_SCALE = (864.0, 576.0)   # default map dims; normalises dx,dy to ~[-1,1]
GOAL_HIDDEN = 128
MOTION_HIDDEN = 32
HEAD_HIDDEN = 256              # per-layer width in the combined head
CNN_FLAT = 4096                # 64 * 8 * 8  (verified below via dummy forward)


def _conv_forward(x, conv1, conv2, conv3):
    x = F.relu(conv1(x))
    x = F.relu(conv2(x))
    x = F.relu(conv3(x))
    return x.flatten(1)


class _CNNBase(nn.Module):
    """Shared CNN + goal-encoder + motion-encoder scaffolding."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 96, 96)
            self._conv_flat = _conv_forward(dummy, self.conv1, self.conv2, self.conv3).shape[1]

        self.goal_scale = nn.Parameter(
            torch.tensor(GOAL_SCALE, dtype=torch.float32), requires_grad=False)
        self.goal_enc1 = nn.Linear(2, GOAL_HIDDEN)
        self.goal_enc2 = nn.Linear(GOAL_HIDDEN, GOAL_HIDDEN)

        self.motion_enc = nn.Linear(4, MOTION_HIDDEN)

    @property
    def feature_dim(self):
        return self._conv_flat + GOAL_HIDDEN + MOTION_HIDDEN

    def _extract(self, image, goal, motion):
        img_flat = _conv_forward(image, self.conv1, self.conv2, self.conv3)
        g = goal / self.goal_scale
        g = F.relu(self.goal_enc1(g))
        g = self.goal_enc2(g)
        m = F.relu(self.motion_enc(motion))
        return torch.cat([img_flat, g, m], dim=1)


class Critic(_CNNBase):
    def __init__(self, action_dim, checkpoint_dir='checkpoints', name='sac_critic'):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.name = name
        self.checkpoint_file = os.path.join(checkpoint_dir, f'{name}.pt')

        head_in = self.feature_dim + action_dim
        self.q1_fc1 = nn.Linear(head_in, HEAD_HIDDEN)
        self.q1_fc2 = nn.Linear(HEAD_HIDDEN, HEAD_HIDDEN)
        self.q1_out = nn.Linear(HEAD_HIDDEN, 1)

        self.q2_fc1 = nn.Linear(head_in, HEAD_HIDDEN)
        self.q2_fc2 = nn.Linear(HEAD_HIDDEN, HEAD_HIDDEN)
        self.q2_out = nn.Linear(HEAD_HIDDEN, 1)

        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight, gain=1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, image, goal, motion, action):
        feat = self._extract(image, goal, motion)
        x = torch.cat([feat, action], dim=1)
        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)

        q2 = F.relu(self.q2_fc1(x))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_out(q2)
        return q1, q2

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, weights_only=True))


class Policy(_CNNBase):
    def __init__(self, action_dim, action_space=None, checkpoint_dir='checkpoints', name='sac_policy'):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.name = name
        self.checkpoint_file = os.path.join(checkpoint_dir, f'{name}.pt')

        self.fc1 = nn.Linear(self.feature_dim, HEAD_HIDDEN)
        self.fc2 = nn.Linear(HEAD_HIDDEN, HEAD_HIDDEN)
        self.mean_fc = nn.Linear(HEAD_HIDDEN, action_dim)
        self.log_std_fc = nn.Linear(HEAD_HIDDEN, action_dim)

        if action_space is not None:
            high = torch.FloatTensor(action_space.high)
            low  = torch.FloatTensor(action_space.low)
            self.action_scale = nn.Parameter((high - low) / 2.0, requires_grad=False)
            self.action_bias  = nn.Parameter((high + low) / 2.0, requires_grad=False)
        else:
            self.action_scale = nn.Parameter(torch.ones(action_dim), requires_grad=False)
            self.action_bias  = nn.Parameter(torch.zeros(action_dim), requires_grad=False)

        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight, gain=1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, image, goal, motion):
        feat = self._extract(image, goal, motion)
        x = F.relu(self.fc1(feat))
        x = F.relu(self.fc2(x))
        mean = self.mean_fc(x)
        log_std = self.log_std_fc(x).clamp(LOG_SIG_MIN, LOG_SIG_MAX)
        return mean, log_std

    def sample(self, image, goal, motion):
        mean, log_std = self.forward(image, goal, motion)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, weights_only=True))
