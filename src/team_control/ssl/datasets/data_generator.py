import csv
import os
from datetime import datetime
import torch


class DataWriter:
    """
    Generic dataset writer for supervised PPO training.
    Stores:
        robot      -> [6]
        ball       -> [2]
        teammates  -> [N,6]
        opponents  -> [M,6]
        action     -> [5]  (vx, vy, w, kick, dribble)
    """

    def __init__(self, name, save_dir="datasets"):
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        self.path = os.path.join(save_dir, f"{name}_{timestamp}.csv")

        self.csv_file = open(self.path, mode="w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.csv_file)

        # Header
        header = [
            "robot",
            "ball",
            "teammates",
            "opponents",
            "vx", "vy", "w", "kick", "dribble"
        ]
        self.writer.writerow(header)
        self.csv_file.flush()

    def _serialize(self, x):
        """
        Converts tensors or lists into flat python lists
        """
        if isinstance(x, torch.Tensor):
            return x.flatten().tolist()
        if isinstance(x, list):
            out = []
            for e in x:
                out.extend(self._serialize(e))
            return out
        return list(x)

    def write(self, robot, ball, teammates, opponents, action):
        """
        robot: Tensor [8]
        ball: Tensor [4]
        teammates: Tensor [N,8] or None
        opponents: Tensor [M,8] or None
        action: (vx, vy, w, kick, dribble)
        """

        row = [
            self._serialize(robot),
            self._serialize(ball),
            self._serialize(teammates),
            self._serialize(opponents),
            action[0], action[1], action[2], action[3], action[4]
        ]

        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()
