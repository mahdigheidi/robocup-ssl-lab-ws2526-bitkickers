from multiprocessing import Process, Queue, Event, set_start_method
import time
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from team_control.ssl.vision.process import vision_worker
from team_control.world.model_manager import WorldModelManager
from team_control.world.model import WorldModel
from team_control.world.model_runner import wm_runner
from team_control.ssl.rl.encoder import Encoder
from team_control.ssl.rl.env import GrSimEnv
from team_control.ssl.rl.referee import Referee, referee_worker
from team_control.ssl.rl.field import Field
from team_control.ssl.manual_controller.goalie_process import goalie_process


def _sample_action(mu, std):
    dist = Normal(mu, std)
    action = dist.sample()
    log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
    entropy = dist.entropy().sum(dim=-1, keepdim=True)
    return action, log_prob, entropy


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        # Relative distance of the ego robot to each of the Ball, YGoalie, YP1, YP2, YP3, BGoalie, BP1, BP2, BP3
        self.memory = nn.LSTM(22, 256, num_layers=1)
        self.backbone = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.mu_head = nn.Linear(128, 5)
        self.log_std = nn.Parameter(torch.zeros(5))

    def init_hidden(self):
        # shape = (sequence_length, ...)
        h = torch.zeros(1, 1, 256)
        c = torch.zeros(1, 1, 256)
        return (h, c)

    def forward(self, observations, hidden_obs_agents):
        actions = []
        log_probs = []
        entropies = []
        next_hidden_obs_agents = []

        for i in range(6):
            observation_embedding, next_hidden_obs = self.memory(observations[i].unsqueeze(0).unsqueeze(0), hidden_obs_agents[i])
            x = self.backbone(observation_embedding)
            mu = self.mu_head(x)
            std = torch.exp(self.log_std)
            action, log_prob, entropy = _sample_action(mu, std)
            actions.append(action)
            log_probs.append(log_prob)
            entropies.append(entropy)
            next_hidden_obs_agents.append(next_hidden_obs)
        actions = torch.cat(actions)
        log_probs = torch.cat(log_probs)
        entropies = torch.cat(entropies)

        return actions, log_probs, entropies, next_hidden_obs_agents


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        # input is game state --> Ball (x, y) + 8 Agents (x, y, sin, cos) = 2 + 8 * 4 = 34
        self.memory = nn.LSTM(34, 256, num_layers=1, batch_first=True)
        self.q = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def init_hidden(self):
        # shape = (batch_size=1, sequence_length, hidden_size)
        h = torch.zeros(1, 1, 256)
        c = torch.zeros(1, 1, 256)
        return (h, c)

    def forward(self, game_state, hidden_game_state):
        game_embedding, next_hidden_game = self.memory(game_state.unsqueeze(0).unsqueeze(0), hidden_game_state)
        value = self.q(game_embedding)
        return value, game_embedding, next_hidden_game


# ============================================================
# MAPPO Rollout Buffer
# ============================================================
class MAPPORolloutBuffer:
    def __init__(self):
        self.states = []
        self.observations = []
        self.hiddens_game = []
        self.hiddens_agents = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.logps = []
        self.entropies = []

    def add(self, game_state, observations, hidden_game, hidden_agents, reward_yellow, done, value, log_probs, entropies):
        self.states.append(game_state.detach())
        self.observations.append(observations)
        self.hiddens_game.append((hidden_game[0].detach(), hidden_game[1].detach()))
        self.hiddens_agents.append([(h.detach(), c.detach()) for (h, c) in hidden_agents])
        self.rewards.append(reward_yellow)
        self.dones.append(done)
        self.values.append(value.detach())
        self.logps.append(log_probs.detach())
        self.entropies.append(entropies.detach())


# ============================================================
# GAE
# ============================================================
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values = values + [0]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    returns = [a + v for a, v in zip(advantages, values[:-1])]
    return advantages, returns


def train_model(actor: Actor, critic: Critic, actor_optim: torch.optim, critic_optim: torch.optim, buffer: MAPPORolloutBuffer):
    # hyperparams
    vf_coef = 0.5
    clip = 0.1
    ent_coef = 0.01
    # max_grad_norm = 0.5

    # the advantages and returns are calculated based on yellow (right side playing team) perspective
    advantages, returns = compute_gae(buffer.rewards, buffer.values, buffer.dones)

    # Critic loss
    current_values = []
    for state, hidden_game in zip(buffer.states, buffer.hiddens_game):
        new_value, _, _ = critic(state, hidden_game)
        current_values.append(new_value)
    critic_loss = 0
    for value, ret in zip(current_values, returns):
        critic_loss += vf_coef * F.mse_loss(value, ret)

    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()

    # Actor loss (sum over agents)
    actor_loss = 0
    for i in range(len(buffer.dones)):
        observations = buffer.observations[i]
        hiddens_agents = buffer.hiddens_agents[i]
        old_logps = buffer.logps[i]
        advantage = advantages[i].detach()
        # print("Advantage is: ", advantage)
        agent_advantages = torch.cat([advantage, advantage, advantage, -advantage, -advantage, -advantage])

        _, new_logp, new_entropies, _ = actor(observations, hiddens_agents)

        ratio = torch.exp(new_logp - old_logps)
        # print("Ratio is: ", ratio)
        surr1 = ratio * agent_advantages
        # print("Surr1 is: ", surr1)
        surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * agent_advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = new_entropies.mean()
        actor_loss += policy_loss - ent_coef * entropy_loss

    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()


def mappo_process(wm):
    env = GrSimEnv(wm, '127.0.0.1', 20011, True)

    actor = Actor()
    critic = Critic()
    actor_optim = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=5e-4)
    rollout_buffer = MAPPORolloutBuffer()

    num_episodes = 20
    training_epochs = 5
    episode_max_time = 0.1  # in minutes
    episode_max_length = int(episode_max_time * 60 * 12)
    episode_rewards = []
    episode_reward = 0
    for episode in range(num_episodes):
        actions = torch.zeros(6, 1, 5)
        hidden_obs = [actor.init_hidden() for _ in range(6)]
        hidden_game = critic.init_hidden()

        for _ in range(episode_max_length):
            game_state, observations, reward_yellow, done = env.step(actions)
            episode_reward += reward_yellow
            value, game_embedding, hidden_game = critic(game_state, hidden_game)

            actions, log_probs, entropies, next_hidden_obs = actor(observations, hidden_obs)

            rollout_buffer.add(
                game_state, observations, hidden_game, hidden_obs,
                reward_yellow, done, value, log_probs, entropies)
            hidden_obs = next_hidden_obs

            if done:
                env.reset()
                break

        print(f"Episode {episode + 1} reward:   ", episode_reward)
        episode_rewards.append(episode_reward)
        # every X episodes we update our policy to improve it online using ppo
        if (episode+1) % 5 == 0:
            for _ in range(training_epochs):
                train_model(actor, critic, actor_optim, critic_optim, rollout_buffer)
            rollout_buffer = MAPPORolloutBuffer()


if __name__ == '__main__':
    vision_q = Queue()
    gc_q = Queue()  # GameController queue, not in use

    vision_wkr = Process(target=vision_worker, args=(vision_q, True, 10020,))

    wm_manager = WorldModelManager()
    wm_manager.start()
    wm = wm_manager.WorldModel()  # pylint: disable=no-member
    wmr = Process(target=wm_runner, args=(wm, vision_q, gc_q,))

    yellow_goalie = Process(target=goalie_process, args=(wm, 'B', '127.0.0.1', 20011, 0, True))
    blue_goalie = Process(target=goalie_process, args=(wm, 'B', '127.0.0.1', 20011, 0, False))
    mappo = Process(target=mappo_process, args=(wm,))

    vision_wkr.start()
    wmr.start()
    yellow_goalie.start()
    blue_goalie.start()
    mappo.start()

    mappo.join()
    yellow_goalie.join()
    blue_goalie.join()
    vision_wkr.join()
    wmr.join()
