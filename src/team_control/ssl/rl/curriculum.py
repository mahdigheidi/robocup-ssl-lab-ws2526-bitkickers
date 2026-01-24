import torch


curriculum_stages = [
    {
        'action_mask': [1, 1, 0, 0, 0],
        'division': 'S_Easy',
        'num_us': 1,
        'num_them': 0,
        'goalie': False,
    }, # Level 1
    [], # Level 2
    [], # Level 3
    [], # Level 4
    [], # Level 5
    [], # Level 6
    [], # Level 7
    [], # Level 8
    [], # Level 9
    [], # Level 10
    [], # Level 11
    [], # Level 12
    [], # Level 13
]


# Each curriculum stage 2000 episodes
class Curriculum:
    def __init__(self):
        self.stage = 0
        self.action_mask = [1, 1, 0, 0, 0] # not needed because the action space is learned by SL
        self.division = 'B' # not needed to change because with SL learns the model physics
        self.num_us = 1   # Curriculum Size
        self.num_them = 0 # Curriculum Size
        self.goalie = False
        self.episodes = 3000 # Curriculum strategy and episode lenght
        self.episode_length = 30

    def update(self, avg_last_hundred_rewards):
        if avg_last_hundred_rewards > 0:
            self.stage += 1

    def state(self):
        return curriculum_stages[self.stage]
