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
# Goal encoder widened 128 -> 256 and RE-INJECTED at every head layer (see _run_head):
# the goal vector was ~3% of the 4256-d feature, so the policy under-used it and learned
# a near-goal-independent fixed path (run 348 reached in exactly 17 steps every time,
# only ~14% of configs). Widening + re-injection keeps the goal signal alive deep in the
# head so the policy actually turns toward an arbitrary commanded goal.
GOAL_HIDDEN = 256
MOTION_HIDDEN = 32
# Head depth/width matched to the DQN champion (QModel head_layers=4, fc_hidden=512).
# The champion deliberately needed depth-4 to represent a value field that isn't flat
# far from the goal (train.py: "the root cause of transit cycles"); the original SAC
# port used a 2x256 head, which is both under-capacity for that value field and easily
# normalised into mush. No LayerNorm — the champion ran head_norm=False and was stable;
# LN was a patch for the undersized critic and it killed the Q-spread (run 345).
HEAD_HIDDEN = 512
HEAD_LAYERS = 4
CNN_FLAT    = 4096   # 64 * 8 * 8  (verified via dummy forward)


def _conv_forward(x, conv1, conv2, conv3):
    x = F.relu(conv1(x))
    x = F.relu(conv2(x))
    x = F.relu(conv3(x))
    return x.flatten(1)


def _mlp_head(in_dim, hidden, n_layers, reinject_dim=0):
    """A stack of n_layers Linear(->hidden); caller applies ReLU after each and its own
    output projection. Mirrors QModel's head ModuleList (the depth lever for the coord
    value field). If reinject_dim>0, every layer AFTER the first takes an extra reinject_dim
    inputs (the goal embedding concatenated back in) so the goal can't wash out with depth."""
    layers, d = nn.ModuleList(), in_dim
    for i in range(n_layers):
        layers.append(nn.Linear(d + (reinject_dim if i > 0 else 0), hidden))
        d = hidden
    return layers


def _run_head(layers, x, goal_emb=None):
    """Run an _mlp_head: ReLU after each layer; if goal_emb given, concat it to the input
    of every layer after the first (the re-injection skip connection)."""
    for i, layer in enumerate(layers):
        if i > 0 and goal_emb is not None:
            x = torch.cat([x, goal_emb], dim=1)
        x = F.relu(layer(x))
    return x


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
        """Returns (feature, goal_emb). feature is the concatenated [img, goal, motion]
        as before; goal_emb is returned separately so the heads can re-inject it at depth."""
        img_flat = _conv_forward(image, self.conv1, self.conv2, self.conv3)
        g = goal / self.goal_scale
        g = F.relu(self.goal_enc1(g))
        g = self.goal_enc2(g)
        m = F.relu(self.motion_enc(motion))
        return torch.cat([img_flat, g, m], dim=1), g

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

    def __init__(self, n_actions, checkpoint_dir='checkpoints', name='sac_critic',
                 head_layers=HEAD_LAYERS, head_hidden=HEAD_HIDDEN):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.name = name
        self.checkpoint_file = os.path.join(checkpoint_dir, f'{name}.pt')

        # Double-Q: two independent head_layers-deep MLP heads over the shared feature,
        # with the goal embedding re-injected at every layer after the first.
        self.q1 = _mlp_head(self.feature_dim, head_hidden, head_layers, reinject_dim=GOAL_HIDDEN)
        self.q1_out = nn.Linear(head_hidden, n_actions)
        self.q2 = _mlp_head(self.feature_dim, head_hidden, head_layers, reinject_dim=GOAL_HIDDEN)
        self.q2_out = nn.Linear(head_hidden, n_actions)

        self.apply(self._weights_init)

    def forward(self, image, goal, motion):
        feat, g = self._extract(image, goal, motion)
        q1 = self.q1_out(_run_head(self.q1, feat, g))   # (B, n_actions)
        q2 = self.q2_out(_run_head(self.q2, feat, g))   # (B, n_actions)
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

    def __init__(self, n_actions, checkpoint_dir='checkpoints', name='sac_policy',
                 head_layers=HEAD_LAYERS, head_hidden=HEAD_HIDDEN):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.name = name
        self.checkpoint_file = os.path.join(checkpoint_dir, f'{name}.pt')

        # Same head depth/width as the critic (symmetric), 4x512 + goal re-injection.
        self.fc = _mlp_head(self.feature_dim, head_hidden, head_layers, reinject_dim=GOAL_HIDDEN)
        self.out = nn.Linear(head_hidden, n_actions)

        self.apply(self._weights_init)

    def forward(self, image, goal, motion):
        feat, g = self._extract(image, goal, motion)
        logits = self.out(_run_head(self.fc, feat, g))
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
