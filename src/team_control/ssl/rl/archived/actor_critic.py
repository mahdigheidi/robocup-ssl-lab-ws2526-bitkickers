import torch
import torch.nn as nn
import torch.nn.functional as F

from team_control.ssl.rl.twohot import two_hot_decode
from team_control.ssl.rl.archived.encoder import Encoder


EMBEDDING_SIZE = 256


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.backbone = nn.Sequential(
            nn.Linear(EMBEDDING_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.vx_head = nn.Linear(256, 11)
        self.vy_head = nn.Linear(256, 11)
        self.w_head = nn.Linear(256, 11)
        self.kick_head = nn.Linear(256, 11)
        self.dribble_head = nn.Linear(256, 11)

    def forward(self, robot, ball, teammates, opponents):
        embedding = self.encoder(robot, ball, teammates, opponents)
        h = self.backbone(embedding)

        vx_dist = F.softmax(self.vx_head(h), dim=-1)
        vy_dist = F.softmax(self.vy_head(h), dim=-1)
        w_dist = F.softmax(self.w_head(h), dim=-1)
        kick_dist = F.softmax(self.kick_head(h), dim=-1)
        dribble_dist = F.softmax(self.dribble_head(h), dim=-1)

        return vx_dist, vy_dist, w_dist, kick_dist, dribble_dist

    def get_action(self, robot, ball, teammates, opponents, noise_std):
        print(robot.shape)
        print(ball.shape)
        print(teammates.shape)
        print(opponents.shape)
        vx_dist, vy_dist, w_dist, kick_dist, dribble_dist = self.forward(robot, ball, teammates, opponents)

        vx = two_hot_decode(vx_dist, 11, -5, 5)
        vy = two_hot_decode(vy_dist, 11, -5, 5)
        w = two_hot_decode(w_dist, 11, -10, 10)
        kick = two_hot_decode(kick_dist, 11, 0, 10)
        dribble = two_hot_decode(dribble_dist, 11, 0, 1)

        action = torch.stack([vx, vy, w, kick, dribble], dim=1)

        if noise_std > 0:
            action += noise_std * torch.randn_like(action)

        return action.clamp(
            min=torch.tensor([-5, -5, -10, 0, 0]),
            max=torch.tensor([5, 5, 10, 10, 1]))


class Critic(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.encoder = Encoder()

        self.q1 = nn.Sequential(
            nn.Linear(EMBEDDING_SIZE + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(EMBEDDING_SIZE + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, robot, ball, teammates, opponents, action):
        embedding = self.encoder(robot, ball, teammates, opponents)
        x = torch.cat([embedding, action], dim=1)
        return self.q1(x), self.q2(x)
