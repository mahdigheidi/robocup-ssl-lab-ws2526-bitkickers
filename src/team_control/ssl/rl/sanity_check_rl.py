from multiprocessing import Process, Queue, Event, set_start_method
import time
import traceback
from collections import deque
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli, Categorical

from team_control.network.robot_command import RobotCommand
from team_control.network.sender import GrSimSender
from team_control.ssl.vision.process import vision_worker
from team_control.world.model_manager import WorldModelManager
from team_control.world.model import WorldModel
from team_control.world.model_runner import wm_runner
from team_control.ssl.rl.archived.encoder import Encoder
from team_control.ssl.rl.env import GrSimEnv
from team_control.ssl.rl.referee import Referee, referee_worker
from team_control.ssl.rl.field import Field
from team_control.ssl.manual_controller.goalie_process import goalie_process


# our model will be trained to play on the right side (yellow) perspective
# if we want to play on the left side (blue) perspective
# we use mirroring logic and flip the x coordinates of all robots and the ball
# ============================================================
# Mirroring logic
# ============================================================
def get_mirrored_state(state):
    """
    Mirrors the global game state for the opposing team.
    Input state shape: [18]
    Structure: Ball(x, y) + 4 Agents(x, y, sin, cos)
    """
    mirrored_state = state.clone()

    # 1. Flip Ball X, Y
    mirrored_state[0:2] *= -1

    # 2. Flip Agent Positions and Headings
    # Each agent has 4 values: x, y, sin, cos
    # Indices 2 to 33 are the 8 agents
    # yellow goalie: 2-5, yellow p1: 6-9, yellow p2: 10-13, yellow p3: 14-17
    # blue goalie: 18-21, blue p1: 22-25, blue p2: 26-29, blue p3: 30-33
    # we must swap yellow and blue agents as well
    # yellow goalie <-> blue goalie
    mirrored_state[2:6], mirrored_state[10:14] = state[10:14] * \
        torch.tensor([-1, -1, -1, -1]), state[2:6] * torch.tensor([-1, -1, -1, -1])
    # yellow p1 <-> blue p1
    mirrored_state[6:10], mirrored_state[14:18] = state[14:18] * \
        torch.tensor([-1, -1, -1, -1]), state[6:10] * torch.tensor([-1, -1, -1, -1])
    # print("Mirrored State shape: ", mirrored_state.shape)
    return mirrored_state


def get_mirrored_obs(observation):
    """
    Mirrors individual agent observations.
    observations shape: [6, 22]
    (Each agent's relative distances/vectors)
    """
    # observation structure:
    # Agent absolute (x, y, sin, cos) = 4
    # Relative to Ball (x, y) = 2
    # Relative to Yellow(Goalie, P1, P2, P3) (x, y) = 8
    # Relative to Blue(Goalie, P1, P2, P3) (x, y) = 8
    obs_mirrored = observation.clone()
    # mirror the ball
    obs_mirrored[4:6] *= -1
    # first swap the relative positions of yellow and blue robots
    obs_mirrored[6:10], obs_mirrored[10:14] = observation[10:14] * \
        torch.tensor([-1, -1, -1, -1]), observation[6:10] * \
        torch.tensor([-1, -1, -1, -1])

    return obs_mirrored


def mirror_action(action):
    """
    Flips the actions sampled from the mirrored model back to
    the environment's global coordinate system.
    action: [vx, vy, w, kick, dribble]
    """
    mirrored_action = list(action)
    mirrored_action[0] *= -1  # Flip vx
    mirrored_action[1] *= -1  # Flip vy
    # w, kick, and dribble usually remain the same
    return mirrored_action


def _sample_action(mu, std, dribble_logits, kick_logits):
    cont_dist = Normal(mu, std)
    z = cont_dist.rsample()  # for reparameterization trick (mean + std * N(0,1))
    cont_action = torch.tanh(z)
    cont_logp = cont_dist.log_prob(z) - torch.log(1 - cont_action.pow(2) + 1e-6)
    cont_logp = cont_logp.sum(dim=-1, keepdim=True)
    cont_entropy = cont_dist.entropy().sum(dim=-1, keepdim=True)

    kick_dist = Categorical(logits=kick_logits)
    kick_action = kick_dist.sample()
    kick_logp = kick_dist.log_prob(kick_action)
    kick_entropy = kick_dist.entropy()

    dribble_dist = Bernoulli(logits=dribble_logits)
    dribble_action = dribble_dist.sample()
    dribble_logp = dribble_dist.log_prob(dribble_action)
    dribble_entropy = dribble_dist.entropy().squeeze(-1)

    action = torch.cat([cont_action, kick_action.unsqueeze(0), dribble_action], dim=-1)
    log_prob = cont_logp + kick_logp + dribble_logp
    entropy = cont_entropy + kick_entropy + dribble_entropy
    # print(entropy)

    return action, log_prob, entropy


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        # Absolute Ego data and Relative distance of the ego robot to each entities
        # Absolute = 4
        # Relative to the Ball, Teammate(Goalie, P1), Opponent(Goalie, P1) (2*5 = 10) = 10 ==> 4 + 10 = 14
        # Stacked over 10 time steps = 14 * 10
        self.backbone = nn.Sequential(
            nn.Linear(14*10, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.mu_head = nn.Linear(128, 3)
        self.log_std = nn.Parameter(torch.zeros(3))

        self.kick_head = nn.Linear(128, 11)  # kick power from 0 to 10
        self.dribble_head = nn.Linear(128, 1)  # dribble on/off

    def forward(self, observation_stack: List[torch.Tensor]):
        observations = torch.cat(list(observation_stack), dim=-1)
        x = self.backbone(observations)
        mu = self.mu_head(x)
        std = torch.exp(self.log_std)
        kick_logits = self.kick_head(x)
        dribble_logits = self.dribble_head(x)
        action, log_prob, entropy = _sample_action(mu, std, dribble_logits, kick_logits)

        return action, log_prob, entropy


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        # input is game state --> Ball (x, y) + 4 Agents (x, y, sin, cos) = 2 + 4 * 4 = 18
        # stacked over 10 time steps = 34 * 10
        self.q = nn.Sequential(
            nn.Linear(18 * 10, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, game_state_stack: List[torch.Tensor]):
        # concatenate all these tensor with size [34,] over the stack frames to get [340,]
        # print(len(list(game_state_stack)))
        # for i in range(len(game_state_stack)):
        #     print(game_state_stack[i])
        game_memory = torch.cat(list(game_state_stack))
        value = self.q(game_memory)
        return value


# ============================================================
# MAPPO Rollout Buffer
# ============================================================
class TeamRolloutBuffer:
    def __init__(self):
        self.state_stacks = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, game_state_buffer, rewards, done, value):
        self.state_stacks.append(list(game_state_buffer))
        self.rewards.append(rewards)
        self.dones.append(done)
        self.values.append(value.detach())


class AgentRolloutBuffer:
    def __init__(self):
        self.observation_stacks = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.logps = []
        self.entropies = []

    def add(self, observations_buffer, reward, done, value, log_probs, entropies):
        self.observation_stacks.append(list(observations_buffer))
        self.rewards.append(reward)
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


def train_critic(critic: Critic, critic_optim: torch.optim, buffer: TeamRolloutBuffer):
    vf_coef = 0.5
    _, returns = compute_gae(buffer.rewards, buffer.values, buffer.dones)
    current_values = []
    for state_stack in buffer.state_stacks:
        new_value = critic(state_stack)
        current_values.append(new_value)
    critic_loss = 0
    for value, ret in zip(current_values, returns):
        critic_loss += vf_coef * F.mse_loss(value, ret)

    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()


def train_actor(actor: Actor, actor_optim: torch.optim, buffer: AgentRolloutBuffer):
    clip = 0.1
    ent_coef = 0.01
    advantages, _ = compute_gae(buffer.rewards, buffer.values, buffer.dones)

    # Actor loss
    actor_loss = 0

    for i in range(len(buffer.dones)):
        old_logps_y = buffer.logps[i]

        _, new_logp_y, new_entropies_y = actor(buffer.observation_stacks[i])

        ratio = torch.exp(new_logp_y - old_logps_y)
        surr1 = ratio * advantages[i]
        surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages[i]
        policy_loss = -torch.min(surr1, surr2)
        actor_loss += policy_loss - ent_coef * new_entropies_y

    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()


def mappo_process(wm, action_p1_y: Queue, action_p1_b: Queue):
    env = GrSimEnv(wm)

    actor_pool = deque(maxlen=5)
    critic_pool = deque(maxlen=5)
    actor = Actor()
    opp_actor = Actor()
    # set the yellow and blue actors to the same initial weights
    opp_actor.load_state_dict(actor.state_dict())
    critic = Critic()
    actor_optim = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=3e-4)

    rollout_yellow = TeamRolloutBuffer()
    rollout_blue = TeamRolloutBuffer()
    rollout_yp1 = AgentRolloutBuffer()
    rollout_bp1 = AgentRolloutBuffer()

    yellow_game_stack = deque(maxlen=10)
    yellow_game_stack.extend([torch.zeros(18) for _ in range(10)])
    blue_game_stack = deque(maxlen=10)
    blue_game_stack.extend([torch.zeros(18) for _ in range(10)])

    obs_y_p1_stack = deque(maxlen=10)
    obs_y_p1_stack.extend([torch.zeros(14) for _ in range(10)])
    obs_b_p1_stack = deque(maxlen=10)
    obs_b_p1_stack.extend([torch.zeros(14) for _ in range(10)])

    training_epochs = 5
    episode_max_time = 0.1  # in minutes
    episode_max_length = int(episode_max_time * 60 * 12)
    episode_rewards_yellow = []
    episode_rewards_blue = []
    episode = 1
    start_time = time.time()
    episode_reward_yellow = 0
    episode_reward_blue = 0
    while True:
        game_state, observations, reward_agents, reward_yellow, reward_blue, done = env.observe()
        mirror_game_state = get_mirrored_state(game_state)

        yellow_game_stack.append(game_state.detach())
        blue_game_stack.append(mirror_game_state.detach())

        obs_y_p1_stack.append(observations[0].detach())
        obs_b_p1_stack.append(get_mirrored_obs(observations[1].detach()))

        episode_reward_yellow += reward_yellow
        episode_reward_blue += reward_blue

        value_yellow = critic(yellow_game_stack)
        value_blue = critic(blue_game_stack)
        action_p1y, log_probs_p1y, entropies_p1y = actor(obs_y_p1_stack)
        action_p1b, log_probs_p1b, entropies_p1b = opp_actor(obs_b_p1_stack)

        action_p1_y.put(action_p1y.squeeze(0).detach().numpy())
        action_p1_b.put(mirror_action(action_p1b.squeeze(0).detach().numpy()))

        rollout_yellow.add(yellow_game_stack, reward_yellow, done, value_yellow)
        rollout_blue.add(blue_game_stack, reward_blue, done, value_blue)
        rollout_yp1.add(obs_y_p1_stack, reward_agents[0]+reward_yellow, done, value_yellow, log_probs_p1y, entropies_p1y)
        rollout_bp1.add(obs_b_p1_stack, reward_agents[1]+reward_blue, done, value_blue, log_probs_p1b, entropies_p1b)

        if done or time.time() - start_time > 10:
            start_time = time.time()
            episode += 1
            print(f"Episode {episode + 1} reward yellow:   ", episode_reward_yellow)
            print(f"Episode {episode + 1} reward blue:     ", episode_reward_blue)
            episode_rewards_yellow.append(episode_reward_yellow)
            episode_rewards_blue.append(episode_reward_blue)
            episode_reward_blue = 0
            episode_reward_yellow = 0
            env.reset()
            # every X episodes we update our policy to improve it online using ppo
            if (episode+1) % 5 == 0:
                print("Training on collected rollouts...")
                for _ in range(training_epochs):
                    train_critic(critic, critic_optim, rollout_yellow)
                    train_critic(critic, critic_optim, rollout_blue)
                    train_actor(actor, actor_optim, rollout_yp1)
                    train_actor(actor, actor_optim, rollout_bp1)

                rollout_yp1 = AgentRolloutBuffer()
                rollout_bp1 = AgentRolloutBuffer()
                actor_pool.append(actor.state_dict())
                critic_pool.append(critic.state_dict())
                # choose one actor from the pool to be the opponent model
                k = min(5, len(actor_pool))
                idx = torch.randint(0, k, (1,)).item()
                opp_actor.load_state_dict(actor_pool[idx])
                if (episode+1) % 50 == 0:
                    torch.save(actor.state_dict(), f'mappo_actor_episode_{episode+1}.pth')
                    torch.save(critic.state_dict(), f'mappo_critic_episode_{episode+1}.pth')


def agent_process(action_queue: Queue, is_yellow: bool, robot_id: int):
    grsim_sender = GrSimSender(ip='127.0.0.1', port=20011, is_yellow=is_yellow)
    while True:
        try:
            action = action_queue.get()
            vx, vy, w, kick, dribble = action

            vx *= 5
            vy *= 5
            w *= 10

            cmd = RobotCommand(
                robot_id=robot_id,
                vx=vx,
                vy=vy,
                w=w,
                kick=kick,
                dribble=dribble,
            )
            # print("Velocity scale: ", (vx**2 + vy**2)**0.5)
            grsim_sender.send_command(cmd)

        except Exception as e:
            print(f"Error in agent_process for robot {robot_id} {'Yellow' if is_yellow else 'Blue'}: {e}")
            traceback.print_exc()


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

    action_p1_y = Queue()
    action_p1_b = Queue()
    agent_y1 = Process(target=agent_process, args=(action_p1_y, True, 1))
    agent_b1 = Process(target=agent_process, args=(action_p1_b, False, 1))
    mappo = Process(target=mappo_process, args=(wm, action_p1_y, action_p1_b))

    vision_wkr.start()
    time.sleep(2)  # give some time for the vision worker to start and populate the world model
    wmr.start()

    mappo.start()
    yellow_goalie.start()
    agent_y1.start()
    blue_goalie.start()
    agent_b1.start()

    mappo.join()
    yellow_goalie.join()
    blue_goalie.join()
    vision_wkr.join()
    wmr.join()
