import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseModel


class QModel(BaseModel):
    def __init__(self, action_dim, input_shape=(3, 96, 96), goal_dim=4,
                 goal_scale=(864.0, 576.0, 864.0, 576.0),
                 goal_hidden=128, fc_hidden=512,
                 goal_layers=1, head_layers=1, bottleneck=None):
        super(QModel, self).__init__()

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

        # Head: head_layers hidden layers (ReLU after each), then output. This is
        # the compositional reasoning block — the depth lever for the coord rep.
        h_layers, in_dim = [], flat_size + goal_hidden
        for _ in range(head_layers):
            h_layers.append(nn.Linear(in_dim, fc_hidden))
            in_dim = fc_hidden
        self.head = nn.ModuleList(h_layers)

        # Optional information bottleneck: squeeze the fused state+goal decision
        # representation through a narrow channel right before the action head, so
        # the net can't keep a high-res (state, goal) -> Q lookup. Targets the
        # coord-memorization (train>>eval) gap while leaving conv perception and
        # head depth intact. None = no bottleneck (== prior coords-deep).
        self.bottleneck = bottleneck
        if bottleneck is not None:
            self.bottleneck_fc = nn.Linear(fc_hidden, bottleneck)
            self.output = nn.Linear(bottleneck, action_dim)
        else:
            self.output = nn.Linear(fc_hidden, action_dim)

        self.apply(self._weights_init)

        print(f"QModel: input={input_shape}, conv_flat={flat_size}, goal_dim={goal_dim}, "
              f"goal_layers={goal_layers}, head_layers={head_layers}, "
              f"bottleneck={bottleneck}, actions={action_dim}")

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

    def forward(self, obs, goal):
        x = self._conv_forward(obs)
        g = self.encode_goal(goal)
        x = torch.cat([x, g], dim=1)
        for layer in self.head:
            x = F.relu(layer(x))
        if self.bottleneck is not None:
            x = F.relu(self.bottleneck_fc(x))
        return self.output(x)

    def _weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
