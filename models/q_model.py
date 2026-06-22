import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseModel


class QModel(BaseModel):
    def __init__(self, action_dim, input_shape=(3, 96, 96), goal_dim=4,
                 goal_scale=(864.0, 576.0, 864.0, 576.0),
                 goal_hidden=128, fc_hidden=512,
                 goal_layers=1, head_layers=1, head_norm=False,
                 use_motion=False, motion_in_dim=None, motion_hidden=32,
                 motion_window=1, macro_h=1, n_base=None):
        super(QModel, self).__init__()
        self.head_norm = head_norm
        self.use_motion = use_motion
        # Stored so eval can rebuild the matching MotionState off the checkpoint.
        self.motion_window = motion_window
        # Macro-action head: the output is over length-macro_h sequences of n_base
        # base actions (action_dim == n_base ** macro_h). Stored so every eval path
        # can decode an output index back into base actions off the checkpoint.
        # macro_h=1 == the original per-step head (n_base == action_dim).
        self.macro_h = macro_h
        self.n_base = n_base if n_base is not None else action_dim

        # Coordinate reframing: goals arrive as absolute coords
        # [robot_x, robot_y, goal_x, goal_y] in raw map pixels (default map
        # 864x576). Scale each component to [-1, 1] so the goal encoder sees the
        # same input range as the obs branch.
        assert len(goal_scale) == goal_dim, "goal_scale must have one entry per goal dim"
        self.register_buffer("goal_scale", torch.tensor(goal_scale, dtype=torch.float32))

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            flat_size = self._conv_forward(dummy).shape[1]

        # Goal encoder: goal_layers linear layers (ReLU between, none after the
        # last so depth-1 reproduces the original single-Linear encoder).
        g_layers, in_dim = [], goal_dim
        for _ in range(goal_layers):
            g_layers.append(nn.Linear(in_dim, goal_hidden))
            in_dim = goal_hidden
        self.goal_encoder = nn.ModuleList(g_layers)

        # Previous-motion encoder (anti-oscillation): its own small projection of
        # [one-hot(last_action) | velocity], fused into the head alongside the goal.
        self.motion_encoder = None
        motion_feat = 0
        if use_motion:
            if motion_in_dim is None:
                from motion import motion_dim
                # motion one-hot is over BASE actions, not macros.
                motion_in_dim = motion_dim(self.n_base, motion_window)
            self.motion_encoder = nn.Linear(motion_in_dim, motion_hidden)
            motion_feat = motion_hidden

        # Head: head_layers hidden layers (ReLU after each), then output. This is
        # the compositional reasoning block — the depth lever for the coord rep.
        h_layers, in_dim = [], flat_size + goal_hidden + motion_feat
        for _ in range(head_layers):
            h_layers.append(nn.Linear(in_dim, fc_hidden))
            in_dim = fc_hidden
        self.head = nn.ModuleList(h_layers)
        # LayerNorm after each head Linear (before ReLU): stabilizes value
        # learning, curbs overestimation, and closes the train/eval gap without
        # the bootstrap-noise problem naked dropout causes. None == off.
        self.head_norms = (nn.ModuleList([nn.LayerNorm(fc_hidden) for _ in range(head_layers)])
                           if head_norm else None)
        self.output = nn.Linear(fc_hidden, action_dim)

        self.apply(self._weights_init)

        print(f"QModel: input={input_shape}, conv_flat={flat_size}, goal_dim={goal_dim}, "
              f"goal_layers={goal_layers}, head_layers={head_layers}, "
              f"head_norm={head_norm}, actions={action_dim}")

    def _conv_forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.flatten(1)

    def encode_goal(self, goal):
        """Scale raw coords to [-1, 1] and encode. All goal encoding goes through
        here so the scaling can't be bypassed. ReLU between layers but not after
        the final one (depth-1 == original single Linear)."""
        g = goal / self.goal_scale
        last = len(self.goal_encoder) - 1
        for i, layer in enumerate(self.goal_encoder):
            g = layer(g)
            if i < last:
                g = F.relu(g)
        return g

    def forward(self, obs, goal, motion=None):
        x = self._conv_forward(obs)
        g = self.encode_goal(goal)
        parts = [x, g]
        if self.motion_encoder is not None:
            assert motion is not None, "use_motion model called without a motion vector"
            parts.append(F.relu(self.motion_encoder(motion)))
        x = torch.cat(parts, dim=1)
        for i, layer in enumerate(self.head):
            x = layer(x)
            if self.head_norms is not None:
                x = self.head_norms[i](x)
            x = F.relu(x)
        return self.output(x)

    def _weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
