"""Discrete SAC actor/critic for HomeBotGoalEnv (discrete action mode).

Architecture: separate CNN actor + CNN double-Q critic, no weight sharing.
Both inherit _CNNBase (same conv stack as DQN champion QModel).

Key difference from continuous SAC:
  - DiscreteQNet:  takes (image, goal, motion) → (q1, q2) each (B, n_actions)
                   NO action input; outputs Q for every action simultaneously
  - DiscretePolicy: takes (image, goal, motion) → (probs, log_probs) each (B, n_actions)
                    categorical distribution, no reparameterisation needed

Discrete SAC update rule (Christodoulou 2019):
  V(s') = Σ_a π(a|s') [Q_target(s', a) - α log π(a|s')]   (exact expectation)
  L_critic = MSE(Q(s, a_taken), r + γ V(s'))               (per taken action)
  L_actor  = Σ_a π(a|s)  [α log π(a|s) - min_Q(s, a)]     (full expectation)
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_SIG_MIN = -4   # kept for reference, not used in discrete policy

GOAL_SCALE  = (864.0, 576.0)
GOAL_HIDDEN = 128
MOTION_HIDDEN = 32
HEAD_HIDDEN = 256
CNN_FLAT    = 4096   # 64 * 8 * 8  (verified via dummy forward)


def _conv_forward(x, conv1, conv2, conv3):
    x = F.relu(conv1(x))
    x = F.relu(conv2(x))
    x = F.relu(conv3(x))
    return x.flatten(1)


class _CNNBase(nn.Module):
    """Shared CNN + goal-encoder + motion-encoder scaffolding (separate weights per subclass)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,  32, kernel_size=8, stride=4)
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

    @staticmethod
    def _weights_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight, gain=1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class DiscreteQNet(_CNNBase):
    """Double-Q critic: outputs Q-values for ALL discrete actions.

    forward() takes NO action input — returns vectors (B, n_actions) for both heads.
    Critic loss indexes into the vector with the taken action via .gather().
    """

    def __init__(self, n_actions, checkpoint_dir='checkpoints', name='sac_critic'):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.name = name
        self.checkpoint_file = os.path.join(checkpoint_dir, f'{name}.pt')

        # LayerNorm after each critic hidden layer (before ReLU). This is the standard
        # cure for value overestimation: it constrains the pre-activation distribution
        # so the critic can't extrapolate to extreme Q on out-of-distribution inputs,
        # bounding the deadly-triad divergence (runs 342/343: mean_q->200, loss->1e6)
        # WITHOUT shrinking the discount/reward signal the way lowering gamma did
        # (run 344 killed learning). Applied only to the critic, not the policy.
        self.q1_fc1 = nn.Linear(self.feature_dim, HEAD_HIDDEN)
        self.q1_ln1 = nn.LayerNorm(HEAD_HIDDEN)
        self.q1_fc2 = nn.Linear(HEAD_HIDDEN, HEAD_HIDDEN)
        self.q1_ln2 = nn.LayerNorm(HEAD_HIDDEN)
        self.q1_out = nn.Linear(HEAD_HIDDEN, n_actions)

        self.q2_fc1 = nn.Linear(self.feature_dim, HEAD_HIDDEN)
        self.q2_ln1 = nn.LayerNorm(HEAD_HIDDEN)
        self.q2_fc2 = nn.Linear(HEAD_HIDDEN, HEAD_HIDDEN)
        self.q2_ln2 = nn.LayerNorm(HEAD_HIDDEN)
        self.q2_out = nn.Linear(HEAD_HIDDEN, n_actions)

        self.apply(self._weights_init)

    def forward(self, image, goal, motion):
        feat = self._extract(image, goal, motion)
        q1 = F.relu(self.q1_ln1(self.q1_fc1(feat)))
        q1 = F.relu(self.q1_ln2(self.q1_fc2(q1)))
        q1 = self.q1_out(q1)   # (B, n_actions)

        q2 = F.relu(self.q2_ln1(self.q2_fc1(feat)))
        q2 = F.relu(self.q2_ln2(self.q2_fc2(q2)))
        q2 = self.q2_out(q2)   # (B, n_actions)
        return q1, q2

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, weights_only=True))


class DiscretePolicy(_CNNBase):
    """Categorical actor: outputs action probabilities over n_actions.

    forward() returns (probs, log_probs) each (B, n_actions).
    log_probs from log_softmax is numerically stable (avoids log(0)).
    get_action() samples or argmaxes for exploration / evaluation.
    """

    def __init__(self, n_actions, checkpoint_dir='checkpoints', name='sac_policy'):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.name = name
        self.checkpoint_file = os.path.join(checkpoint_dir, f'{name}.pt')

        self.fc1 = nn.Linear(self.feature_dim, HEAD_HIDDEN)
        self.fc2 = nn.Linear(HEAD_HIDDEN, HEAD_HIDDEN)
        self.out = nn.Linear(HEAD_HIDDEN, n_actions)

        self.apply(self._weights_init)

    def forward(self, image, goal, motion):
        feat = self._extract(image, goal, motion)
        x = F.relu(self.fc1(feat))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        probs     = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs

    def get_action(self, image, goal, motion, evaluate=False):
        """Returns (action LongTensor (B,), log_probs (B, n_actions))."""
        probs, log_probs = self.forward(image, goal, motion)
        if evaluate:
            action = probs.argmax(dim=-1)
        else:
            action = torch.multinomial(probs, 1).squeeze(-1)
        return action, log_probs

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, weights_only=True))
