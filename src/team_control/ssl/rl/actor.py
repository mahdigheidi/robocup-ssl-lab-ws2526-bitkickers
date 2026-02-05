import torch

import torch.nn as nn


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(50, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.motion_head = nn.Sequential(
            nn.Linear(256, 3)
        )
        self.kick_head = nn.Sequential(
            nn.Linear(256, 1)
        )
        self.dribble_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.net(x)

        motion = self.motion_head(h)
        kick = self.kick_head(h)
        dribble = self.dribble_head(h)

        # Concatenate along the action dimension
        # action = torch.cat([vx, vy, w, kick, dribble], dim=-1)

        return motion, kick, dribble
