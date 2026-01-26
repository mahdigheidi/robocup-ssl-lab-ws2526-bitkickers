import csv
import ast
from multiprocessing import Process, Queue

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from team_control.ssl.rl.encoder import Encoder
from team_control.ssl.rl.env import FootballEnv
from team_control.ssl.rl.callback import TrainingCallback
from team_control.ssl.vision.process import vision_worker
from team_control.utils.visualization import plot_rewards
from team_control.world.model_manager import WorldModelManager
from team_control.world.model_runner import wm_runner
from team_control.ssl.rl.curriculum import Curriculum


# ============================================================
# PPO Actor-Critic with Encoder + LSTM
# ============================================================

class PPOActorCritic(nn.Module):
    def __init__(self, encoder: Encoder, hidden_size=256):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True
        )

        # Mean head for 5D action
        # instead of regressing the mean values, do two hot encoding on them
        self.mu_head = nn.Linear(hidden_size, 5)

        # Log standard deviation (one parameter per action dimension)
        self.log_std = nn.Parameter(torch.zeros(5))

        # Critic head
        self.value_head = nn.Linear(hidden_size, 1)

    def init_hidden(self, device):
        h = torch.zeros(3, 1, self.hidden_size, device=device)
        c = torch.zeros(3, 1, self.hidden_size, device=device)
        return (h, c)

    def forward(self, robot, ball, team, opp, hidden):
        """
        robot: [1, 6]
        ball:  [1, 2]
        team:  Tensor [N, 6] or None
        opp:   Tensor [M, 6] or None
        hidden: (h, c)
        """

        # print(robot.shape, ball.shape, team.shape, opp.shape)

        emb = self.encoder(robot, ball, team, opp)      # [1, 256]
        emb = emb.unsqueeze(1)                          # [1, 1, 256]

        out, hidden = self.lstm(emb, hidden)            # [1, 1, H]
        h = out.squeeze(1)                              # [1, H]

        mu = self.mu_head(h)                     # [1, 5]
        std = torch.exp(self.log_std).unsqueeze(0)  # [1, 5]
        value = self.value_head(h)
        return (mu, std), value, hidden


# ============================================================
# Action sampling
# ============================================================
def sample_action(mu, std):
    dist = torch.distributions.Normal(mu, std)
    action = dist.sample()
    logp = dist.log_prob(action).sum(dim=-1)       # sum over action dimensions
    entropy = dist.entropy().sum(dim=-1)
    return action, logp, entropy


# ============================================================
# Rollout Buffer (On-policy)
# ============================================================
class RolloutBuffer:
    def __init__(self):
        self.states = []        # (robot, ball, team, opp)
        self.actions = []       # action tensors
        self.logps = []         # old log-prob (detached)
        self.values = []        # old value (detached, scalar)
        self.rewards = []
        self.dones = []
        self.entropies = []     # old entropy (detached)

    def clear(self):
        self.__init__()


# ============================================================
# GAE
# ============================================================
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values = values + [0]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    returns = [a + v for a, v in zip(advantages, values[:-1])]
    return advantages, returns


# ============================================================
# PPO Update
# ============================================================
def ppo_update(policy, optimizer, buffer, clip=0.2, vf_coef=0.5, ent_coef=0.01, device="cpu"):
    advantages, returns = compute_gae(
        buffer.rewards, buffer.values, buffer.dones
    )

    advantages = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    returns = torch.as_tensor(returns, dtype=torch.float32, device=device)

    loss = 0.0
    hidden = policy.init_hidden(device)

    for (state, action, logp_old, value_old, adv, ret, done) in zip(
        buffer.states,
        buffer.actions,
        buffer.logps,
        buffer.values,
        advantages,
        returns,
        buffer.dones,
    ):
        robot, ball, team, opp = state
        robot = robot.to(device)
        ball = ball.to(device)
        if team is not None:
            team = team.to(device)
        if opp is not None:
            opp = opp.to(device)

        # Forward again to build a fresh graph
        (mu, std), value, hidden = policy(robot, ball, team, opp, hidden)
        dist = torch.distributions.Normal(mu, std)

        # Log-prob of taken action under current policy
        logp = dist.log_prob(action).sum(dim=-1)

        # PPO ratio
        ratio = torch.exp(logp - logp_old)

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
        policy_loss = -torch.min(surr1, surr2)

        # Value loss
        value_loss = F.mse_loss(value.squeeze(), ret)

        # Entropy bonus
        entropy = dist.entropy().sum(dim=-1)
        entropy_loss = -entropy

        loss = loss + policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

        if done:
            hidden = policy.init_hidden(device)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# ============================================================
# Supervised Learning (Behavior Cloning)
# ============================================================
def supervised_update(policy, optimizer, csv_path, device, epochs=5, batch_size=32):
    """
    Supervised behavior cloning from CSV generated by DataWriter.

    CSV format:
        robot: [8]
        ball: [4]
        teammates: flattened list or []
        opponents: flattened list or []
        vx, vy, w, kick, dribble: floats
    """

    def parse_row(row):
        robot = torch.tensor(ast.literal_eval(row["robot"]), dtype=torch.float32).unsqueeze(0)
        ball = torch.tensor(ast.literal_eval(row["ball"]), dtype=torch.float32).unsqueeze(0)

        team_raw = ast.literal_eval(row["teammates"])
        opp_raw = ast.literal_eval(row["opponents"])

        team = None
        opp = None

        if len(team_raw) > 0:
            team = torch.tensor(team_raw, dtype=torch.float32).view(-1, robot.shape[-1])
        if len(opp_raw) > 0:
            opp = torch.tensor(opp_raw, dtype=torch.float32).view(-1, robot.shape[-1])

        action = torch.tensor(
            [
                float(row["vx"]),
                float(row["vy"]),
                float(row["w"]),
                float(row["kick"]),
                float(row["dribble"]),
            ],
            dtype=torch.float32,
        ).unsqueeze(0)

        return robot, ball, team, opp, action

    # Load CSV
    dataset = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset.append(parse_row(row))

    print(f"[Supervised] Loaded {len(dataset)} samples from {csv_path}")

    policy.train()

    for epoch in range(epochs):
        # np.random.shuffle(dataset)
        total_loss = 0.0

        for start in range(0, len(dataset), batch_size):
            batch = dataset[start:start + batch_size]
            hidden = policy.init_hidden(device)
            loss = 0.0

            for robot, ball, team, opp, target_action in batch:
                robot = robot.to(device)
                ball = ball.to(device)
                if team is not None:
                    team = team.to(device)
                if opp is not None:
                    opp = opp.to(device)
                target_action = target_action.to(device)

                (mu, std), _, hidden = policy(robot, ball, team, opp, hidden)

                # Mean squared error on Gaussian mean
                loss = loss + F.mse_loss(mu, target_action) / batch_size

                # Detach LSTM state between samples
                hidden = (hidden[0].detach(), hidden[1].detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() / (len(dataset) // batch_size)

        print(f"[Supervised] Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")


# ============================================================
# Minimal Training Loop Skeleton
# ============================================================

def train_ppo(env: FootballEnv, policy: PPOActorCritic, optimizer, device, rollout_length, update_epochs=4):

    callback = TrainingCallback(save_dir="ppo_snapshots", save_freq=100)
    # latest_episode, episodes_rewards = callback.load_if_exists(policy)

    buffer = RolloutBuffer()
    num_episodes = 400
    rewards = []
    for episode in tqdm(range(0, num_episodes)):
        hidden = policy.init_hidden(device)
        episode_reward = 0

        state = env.reset()
        done = False

        for _ in range(rollout_length):

            robot, ball, team, opp = state
            robot = robot.to(device)
            ball = ball.to(device)
            team = team.to(device)
            opp = opp.to(device)

            (mu, std), value, hidden = policy(robot, ball, team, opp, hidden)
            actions, logp, entropy = sample_action(mu, std)

            # You decode bins → continuous yourself here
            # continuous_action = decode(actions)

            next_state, reward, done = env.step(actions)
            # print(env.robot_id, env.is_team_yellow, reward)
            episode_reward += reward

            buffer.states.append(state)
            buffer.actions.append(actions.detach())
            buffer.logps.append(logp.detach())
            buffer.values.append(value.squeeze().detach())
            buffer.rewards.append(float(reward))
            buffer.dones.append(bool(done))
            buffer.entropies.append(entropy.detach())

            state = next_state

            if done:
                hidden = policy.init_hidden(device)
                state = env.reset()


        for _ in range(update_epochs):
            ppo_update(policy, optimizer, buffer, device=device)
        rewards.append(episode_reward)
        print(f"Episode: {episode}, Reward: {episode_reward}")
        buffer.clear()
        callback.on_episode_end(episode, policy, rewards)


def run_ppo(env: FootballEnv, policy: PPOActorCritic, device, rollout_length):
    num_episodes = 400
    for _ in range(num_episodes):
        hidden = policy.init_hidden(device)

        state = env.reset()
        done = False

        for _ in range(rollout_length):

            robot, ball, team, opp = state
            robot = robot.to(device)
            ball = ball.to(device)
            team = team.to(device)
            opp = opp.to(device)

            (mu, _), _, hidden = policy(robot, ball, team, opp, hidden)
            # actions, logp, entropy = sample_action(mu, std)
            actions = mu.detach()

            # You decode bins → continuous yourself here
            # continuous_action = decode(actions)

            next_state, _, done = env.step(actions)

            state = next_state

            if done:
                hidden = policy.init_hidden(device)
                state = env.reset()


# ============================================================
# USAGE
# ============================================================
if __name__ == '__main__':
    device = torch.device("cpu")
    curriculum = Curriculum()
    goalie_encoder = Encoder()
    goalie_policy = PPOActorCritic(goalie_encoder).to(device)
    goalie_optimizer = torch.optim.Adam(goalie_policy.parameters(), lr=3e-4)

    player_encoder = Encoder()
    player_policy = PPOActorCritic(player_encoder).to(device)
    player_optimizer = torch.optim.Adam(player_policy.parameters(), lr=3e-4)


    def load_model(policy, optimizer, path):
        device = torch.device("cpu")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        policy.load_state_dict(checkpoint["policy_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    load_model(player_policy, player_optimizer, "./pretrained_player.pt")
    load_model(goalie_policy, goalie_optimizer, "./pretrained_goalie.pt")


    # # PPO - Selfplay:
    vision_q = Queue()
    gc_q = Queue()  # GameController queue, not in use

    vision_wkr = Process(target=vision_worker, args=(vision_q, True, 10020,))

    with WorldModelManager() as wm_manager:
        wm = wm_manager.WorldModel()  # pylint: disable=no-member
        wmr = Process(target=wm_runner, args=(wm, vision_q, gc_q,))

        # todo: make sure it works with the blue as well
        env_yellow_goalie = FootballEnv(
            wm, "127.0.0.1", 20011, 0, True, 0, curriculum, True,
        )
        env_blue_goalie = FootballEnv(
            wm, "127.0.0.1", 20011, 0, False, 0, curriculum, True,
        )
        env_yellow_striker = FootballEnv(
            wm, "127.0.0.1", 20011, 1, True, 1, curriculum, True,
        )
        env_blue_striker = FootballEnv(
            wm, "127.0.0.1", 20011, 1, False, 1, curriculum, True,
        )

        vision_wkr.start()
        wmr.start()

        yellow_goalie = Process(target=train_ppo, args=(env_yellow_goalie, goalie_policy, goalie_optimizer, device, 32,))
        blue_goalie = Process(target=train_ppo, args=(env_blue_goalie, goalie_policy, goalie_optimizer, device, 32,))
        yellow_striker = Process(target=train_ppo, args=(env_yellow_striker, player_policy, player_optimizer, device, 32,))
        blue_striker = Process(target=train_ppo, args=(env_blue_striker, player_policy, player_optimizer, device, 32,))

        yellow_goalie.start()
        blue_goalie.start()
        yellow_striker.start()
        blue_striker.start()


        # # TODO: test it with blue and let blue play as well.
        # ppo_agent = Process(target=train_ppo, args=(env, goalie_policy, optimizer, device, 100))
        # ppo_agent.start()
        # ppo_agent.join()
        yellow_goalie.join()
        blue_goalie.join()
        yellow_striker.join()
        blue_striker.join()

        vision_wkr.join()
        wmr.join()

    # Supervised:
    # for i in range(5):
    #     print("Starting Round ", i)
    #     supervised_update(
    #         player_policy,
    #         player_optimizer,
    #         csv_path="../manual_controller/datasets/yellow_striker.csv",
    #         device=device,
    #         epochs=20,
    #         batch_size=128
    #     )
    #     supervised_update(
    #         player_policy,
    #         player_optimizer,
    #         csv_path="../manual_controller/datasets/blue_striker.csv",
    #         device=device,
    #         epochs=20,
    #         batch_size=128
    #     )
    #     supervised_update(
    #         goalie_policy,
    #         goalie_optimizer,
    #         csv_path="../manual_controller/datasets/yellow_goalie.csv",
    #         device=device,
    #         epochs=20,
    #         batch_size=128
    #     )
    #     supervised_update(
    #         goalie_policy,
    #         goalie_optimizer,
    #         csv_path="../manual_controller/datasets/blue_goalie.csv",
    #         device=device,
    #         epochs=20,
    #         batch_size=128
    #     )
    # checkpoint = {
    #     "policy_state_dict": goalie_policy.state_dict(),
    #     "optimizer_state_dict": goalie_optimizer.state_dict(),
    # }
    # torch.save(checkpoint, "pretrained_goalie.pt")
    # print(f"[PPO] Model saved to pretrained_goalie.pt")

    # checkpoint = {
    #     "policy_state_dict": player_policy.state_dict(),
    #     "optimizer_state_dict": player_optimizer.state_dict(),
    # }
    # torch.save(checkpoint, "pretrained_player.pt")
    # print(f"[PPO] Model saved to pretrained_player.pt")
