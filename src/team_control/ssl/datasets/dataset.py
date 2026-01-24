import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

from team_control.ssl.rl.twohot import two_hot_encode


class FootballDataset(Dataset):
    def __init__(self, csv_path, device):
        self.df = pd.read_csv(csv_path)
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        robot = torch.tensor(np.array(row["Robot"].split("_"), dtype=np.float32), dtype=torch.float32)
        ball = np.array(row["Ball"].split("_"), dtype=np.float32)
        teammates = np.array([0 for i in range(8)], dtype=np.float32) if row.isnull()["Teammates"] else np.array(
            [teammate.split('_') for teammate in row["Teammates"].split('/')], dtype=np.float32)
        opponents = np.array([0 for i in range(8)], dtype=np.float32) if row.isnull()["Opponents"] else np.array(
            [opponent.split('_') for opponent in row["Opponents"].split('/')], dtype=np.float32)

        action = np.array(row["Action"].split("_"), dtype=np.float32)
        action_two_hot = []
        action_two_hot.append(two_hot_encode(action[0], 11, -5, 5))
        action_two_hot.append(two_hot_encode(action[1], 11, -5, 5))
        action_two_hot.append(two_hot_encode(action[2], 11, -10, 10))
        action_two_hot.append(two_hot_encode(action[3], 11, 0, 10))
        action_two_hot.append(two_hot_encode(action[4], 11, 0, 1))

        return robot, ball, teammates, opponents, np.array(action_two_hot, dtype=np.float32)


if __name__ == '__main__':
    ds = FootballDataset("../datasets/generated_data.csv", "cpu")
    robot, ball, teammates, opponents, action = ds[128]
    print("Robot: ")
    print(robot)
    print("Ball: ")
    print(ball)
    print("Teammates: ")
    print(teammates)
    print("Opponents: ")
    print(opponents)
    print("Action: ")
    print(action)
