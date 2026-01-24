from multiprocessing import Process, Queue
import traceback

import torch
import torch.nn.functional as F
from tqdm import tqdm

from team_control.ssl.rl.actor_critic import Actor, Critic
from team_control.ssl.rl.callback import TrainingCallback
# from team_control.ssl.rl.curriculum import Curriculum
from team_control.ssl.rl.encoder import Encoder
from team_control.ssl.rl.env import FootballEnv
from team_control.ssl.offpolicy_rl.replay_buffer import ReplayBuffer
from team_control.ssl.vision.process import vision_worker
from team_control.utils.visualization import plot_rewards
from team_control.ssl.rl.reward import Reward
from team_control.ssl.rl.curriculum import Curriculum
from team_control.world.model_manager import WorldModelManager
from team_control.world.model_runner import wm_runner


class TD3Agent:
    def __init__(self, action_dim, encoder, device):
        self.device = device

        # load the actor
        # actor_model_path = "model_snapshots/actor/best_actor.pt"
        # best_actor_dict = torch.load(actor_model_path)
        # actor = Actor(encoder)
        # actor.load_state_dict(best_actor_dict['model_state_dict'])

        # Shared actor between all agents
        self.actor = Actor().to(device)
        self.actor_target = Actor().to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Shared critic between all agents
        self.critic = Critic(action_dim).to(device)
        self.critic_target = Critic(action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        # self.curriculum = Curriculum()

        # Hyperparams
        self.gamma = 0.99
        self.tau = 0.01
        self.policy_noise = 0.1
        self.noise_clip = 0.2
        self.policy_delay = 3
        self.total_it = 0

    def select_action(self, state, noise):
        robot, ball, teammates, opponents = state
        robot = torch.FloatTensor(robot).unsqueeze(0).to(self.device)
        ball = torch.FloatTensor(ball).unsqueeze(0).to(self.device)
        teammates = torch.FloatTensor(teammates).unsqueeze(0).to(self.device) if teammates is not None else None
        opponents = torch.FloatTensor(opponents).unsqueeze(0).to(self.device) if opponents is not None else None
        action = self.actor.get_action(robot, ball, teammates, opponents, noise).detach().cpu()
        return action

    def train(self, replay_buffer: ReplayBuffer, batch_size=256):
        self.total_it += 1

        s, a, r, s_next, d = replay_buffer.sample(batch_size)
        print(s_next)

        with torch.no_grad():
            robot, ball, teammates, opponents = zip(*s_next)
            print(robot.shape)
            print(ball.shape)
            print(teammates.shape)
            print(opponents.shape)
            exit(0)

            robot = torch.cat(robot, dim=0).to(self.device)   # [B, robot_dim]
            ball  = torch.cat(ball, dim=0).to(self.device)    # [B, ball_dim]


            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_a = self.actor_target.get_action(robot, ball, teammates, opponents, noise_std=0.0) + noise
            q1_t, q2_t = self.critic_target(robot, ball, teammates, opponents, next_a)
            q_target = r + self.gamma * (1 - d) * torch.min(q1_t, q2_t)

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # freeze the encoder after it's completely trained
        # use when you want to freeze the actor weights
        # if self.total_it > 20000 and self.total_it % self.policy_delay == 0:
        if self.total_it % self.policy_delay == 0:
            actor_action = self.actor.get_action(s, noise_std=0.0)
            x = torch.cat([s, actor_action], dim=1)
            actor_loss = -self.critic.q1(x).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)


def train_td3(
    env: FootballEnv,
    curriculum: Curriculum,
    callback: TrainingCallback,
    encoder: Encoder,
):
    device = torch.device("cpu")

    episodes = curriculum.episodes
    max_steps = curriculum.episode_length

    agent = TD3Agent(
        action_dim=5,
        encoder=encoder,
        device=device
    )
    start_episode, rewards = callback.load_if_exists(agent)
    # previous_model = torch.load('/Users/mahdigheidi/Documents/Univ/RoboCupLab/robocup-ssl/src/team_control/ssl/rl/model_snapshots/td3_ep_800.pt')
    # starting_episode = previous_model['episode']
    # agent.actor.load_state_dict(previous_model['actor'])
    # agent.critic.load_state_dict(previous_model['critic'])
    # agent.actor_target.load_state_dict(previous_model['actor_target'])
    # agent.critic_target.load_state_dict(previous_model['critic_target'])
    # rewards = previous_model['rewards']
    buffer = ReplayBuffer()

    for ep in tqdm(range(start_episode, episodes)):
        state = env.reset()
        ep_reward = 0

        for _ in range(max_steps):
            action = agent.select_action(state, noise=0.05)  # is our noise hyperparam good?
            try:
                next_state, reward, done = env.step(action)
                print("len(nextstate): ", len(next_state))
            except Exception as e:
                traceback.print_exc()

            buffer.push(state, action, reward, next_state, done)
            # print("Reward: ", reward)

            state = next_state
            ep_reward += reward

            if done:
                break

        rewards.append(ep_reward)

        if len(buffer) > 300:
            for _ in range(50):
                agent.train(buffer)

        if callback is not None:
            callback.on_episode_end(ep, agent, rewards)

        print(f"Episode {ep:04d} | Return {ep_reward:.2f}")

    return rewards


def td3_process(
        wm, simulator_ip,
        command_port, division,
        robot_id, is_team_yellow, role, encoder,
):
    training = True
    curriculum = Curriculum()
    env = FootballEnv(wm, simulator_ip, command_port, robot_id, is_team_yellow, role, curriculum, training)

    callback = TrainingCallback(save_dir="model_snapshots/", save_freq=100)

    rewards = train_td3(env=env, curriculum=curriculum, callback=callback, encoder=encoder)
    plot_rewards(rewards)


if __name__ == '__main__':
    use_sim = True
    simulator_ip = "127.0.0.1"
    command_port = 20011
    vision_port = 10020
    division = 'B'
    robot_id = 0
    is_team_yellow = True

    vision_q = Queue()
    gc_q = Queue()  # GameController queue, not in use

    vision_wkr = Process(target=vision_worker, args=(vision_q, use_sim, vision_port,))

    with WorldModelManager() as wm_manager:
        wm = wm_manager.WorldModel()  # pylint: disable=no-member
        wmr = Process(target=wm_runner, args=(wm, vision_q, gc_q,))

        vision_wkr.start()
        wmr.start()

        # TODO: test it with blue and let blue play as well.

        # td3_yg = Process(target=td3_process, args=(
        # wm, simulator_ip, command_port, division, robot_id, is_team_yellow, 0,))
        # td3_bg = Process(target=td3_process, args=(
        #     wm, simulator_ip, command_port, division, robot_id, is_team_yellow, 0,))
        # td3_b1 = Process(target=td3_process, args=(
        #     wm, simulator_ip, command_port, division, robot_id, is_team_yellow, 1,))

        encoder = Encoder()
        td3_y1 = Process(target=td3_process, args=(
            wm, simulator_ip, command_port, division, robot_id, is_team_yellow, 0, encoder,))
        td3_y1.start()
        td3_y1.join()

        vision_wkr.join()
        wmr.join()
