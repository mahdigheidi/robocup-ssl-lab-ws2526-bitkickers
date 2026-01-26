from multiprocessing import Process, Queue, Event, set_start_method
import time

from team_control.robot import role
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from team_control.ssl.vision.process import vision_worker
from team_control.world.model_manager import WorldModelManager
from team_control.world.model_runner import wm_runner
from team_control.ssl.rl.encoder import Encoder
from team_control.ssl.rl.env import FootballEnv
from team_control.ssl.rl.referee import Referee
from team_control.ssl.rl.field import Field


# ============================================================
# MAPPO Policy (Shared actor + Centralized critic) per team
# So that means we have a mappo policy for blue and one for yellow
# ============================================================
class MAPPOPolicy(nn.Module):
    def __init__(self, encoder, num_agents, hidden_size=256):
        super().__init__()
        # Team information
        self.num_agents = num_agents
        self.team_spirit = 0.7

        self.encoder = encoder
        self.lstm_layers = 3
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            256,
            hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
        )

        # Actor heads (shared)
        # Mean head for 5D action
        self.mu_head = nn.Linear(hidden_size, 5)

        # Log standard deviation (one parameter per action dimension)
        self.log_std = nn.Parameter(torch.zeros(5))

        # Centralized critic
        self.central_value = nn.Sequential(
            nn.Linear(hidden_size * num_agents, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    # One hidden state per agent
    def init_hidden(self, device):
        # h = torch.zeros(1, self.num_agents, self.hidden_size, device=device)
        h = torch.zeros(self.lstm_layers, 1, self.hidden_size, device=device)
        c = torch.zeros(self.lstm_layers, 1, self.hidden_size, device=device)
        return (h, c)

    def actor_forward(self, robot, ball, team, opp, hidden_state):
        # encoder expects [batch, features], add batch dim if missing
        emb = self.encoder(robot, ball, team, opp).unsqueeze(1)
        out, next_hidden = self.lstm(emb, hidden_state)
        h = out.squeeze(1)

        mu = self.mu_head(h)  # Bound actions to [-1, 1]
        std = torch.exp(self.log_std)
        return mu, std, next_hidden, h

    def critic_forward(self, all_hidden_states):
        # all_hidden_states: [batch, num_agents * hidden_size]
        return self.central_value(all_hidden_states)


# ============================================================
# Action sampling
# ============================================================
def sample_action(mu, std):
    dist = Normal(mu, std)
    action = dist.sample()
    log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
    entropy = dist.entropy().sum(dim=-1, keepdim=True)
    return action, log_prob, entropy


def pack_transition(agent_id, state, action, logp, reward, done, hidden, value):
    # detach hidden states
    if hidden is not None:
        h, c = hidden
        hidden = (h.detach().cpu(), c.detach().cpu())
    # detach value
    if torch.is_tensor(value):
        value = value.detach().cpu()
    return {
        "agent_id": agent_id,
        "state": state,
        "action": action.detach().cpu(),
        "logp": logp.detach().cpu(),
        "reward": float(reward),
        "done": bool(done),
        "hidden": hidden,
        "value": value
    }


# ============================================================
# MAPPO Rollout Buffer
# ============================================================
class MAPPORolloutBuffer:
    def __init__(self, num_agents):
        self.logps = [[] for _ in range(num_agents)]
        self.rewards = []
        self.dones = []
        self.values = []


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


# ============================================================
# MAPPO Update
# ============================================================
def mappo_update(policy, optimizer, buffer, clip=0.2, vf_coef=0.5, ent_coef=0.01):
    advantages, returns = compute_gae(
        buffer.rewards, buffer.values, buffer.dones
    )

    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)

    loss = 0

    # Actor loss (sum over agents)
    for i in range(policy.num_agents):
        for idx, (logp_old, adv) in enumerate(zip(buffer.logps[i], advantages)):
            # Recompute new_logp via forward pass
            # For simplicity, assuming states and hidden states are accessible here
            # This is a placeholder: in a real implementation, you'd store states and hidden states to recompute
            # Here, we just reuse logp_old as new_logp to fix math, but ideally recompute properly
            new_logp = logp_old  # Placeholder

            ratio = torch.exp(new_logp - logp_old)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
            policy_loss = -torch.min(surr1, surr2)
            entropy_loss = 0  # Entropy not stored in buffer anymore, set to 0
            loss += policy_loss + ent_coef * entropy_loss

    # Critic loss
    for value, ret in zip(buffer.values, returns):
        loss += vf_coef * F.mse_loss(value.squeeze(), ret)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def run_agent(halt_event,
              rollout_queue,
              policy_state_queue,
              wm,
              device,
              agent_id,
              is_team_yellow,
              role,
              ):

    env = FootballEnv(wm, "127.0.0.1", 20011, agent_id, is_team_yellow, role, True)

    hidden = None
    local_policy = MAPPOPolicy(Encoder(), num_agents=4).to(device)
    num_agents = 4

    state = env.get_normalized_state()
    while True:

        # Wait for updated policy weights
        # if local_policy is None and not policy_state_queue.empty():
        #     state_dict = policy_state_queue.get()
        #     local_policy = MAPPOPolicy(Encoder(), num_agents=num_agents).to(device)
        #     local_policy.load_state_dict(state_dict)
        #     local_policy.eval()
        #     hidden = local_policy.init_hidden(device)

        if halt_event.is_set():
            time.sleep(0.01)
            continue

        robot, ball, team, opp = state

        robot = robot.to(device)
        ball = ball.to(device)
        team = team.to(device)
        opp = opp.to(device)

        mu, std, hidden, h = local_policy.actor_forward(robot, ball, team, opp, hidden)
        actions, logp, entropy = sample_action(mu, std)
        value = torch.zeros(1) #placeholder
        #value = local_policy.central_value(h).detach().cpu()

        next_state, reward, done = env.step(actions)

        transition = pack_transition(
            agent_id, state, actions, logp, reward, done, hidden, value
        )
        rollout_queue.put(transition)

        state = next_state

        if done:
            hidden = local_policy.init_hidden(device)


def train_model(halt_event, rollout_queue: Queue, policy_state_queue: Queue, num_agents: int):

    device = torch.device("cpu")
    encoder = Encoder()
    policy = MAPPOPolicy(encoder, num_agents).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    policy_state_queue.put(policy.state_dict())
    print("[Trainer] Started")

    buffer = MAPPORolloutBuffer(num_agents)

    while True:

        # Collect transitions
        transition = rollout_queue.get()

        agent_id = transition["agent_id"]
        buffer.logps[agent_id].append(transition["logp"])
        buffer.rewards.append(transition["reward"])
        buffer.dones.append(transition["done"])
        buffer.values.append(transition["value"])

        # Check if referee halted → time to train
        if halt_event.is_set():
            print("[Trainer] Halt received, training...")

            mappo_update(policy, optimizer, buffer)

            buffer = MAPPORolloutBuffer(num_agents)
            halt_event.clear()

            # Send new policy weights to agents
            policy_state_queue.put(policy.state_dict())

            print("[Trainer] Policy updated and broadcasted")


def mappo(wm, halt_event, is_team_yellow):
    print("Starting MAPPO for team", "Yellow" if is_team_yellow else "Blue")
    device = torch.device("cpu")
    num_agents = 4

    rollout_queue = Queue()

    policy_state_queue = Queue()

    #env_goalie = FootballEnv(wm, "127.0.0.1", 20011, 0, 0, True,)
    goalie = Process(
        target=run_agent,
        args=(halt_event, rollout_queue, policy_state_queue, wm, device, 0, is_team_yellow, 0)
    )

    #env_p1 = FootballEnv(wm, "127.0.0.1", 20011, 1, is_team_yellow, 1, True,)
    player_1 = Process(
        target=run_agent,
        args=(halt_event, rollout_queue, policy_state_queue, wm, device, 1, is_team_yellow, 1)
    )

    #env_p2 = FootballEnv(wm, "127.0.0.1", 20011, 2, is_team_yellow, 1, True,)
    player_2 = Process(
        target=run_agent,
        # args=(halt_event, rollout_queue, env_p2, device, 2)
        args=(halt_event, rollout_queue, policy_state_queue, wm, device, 2, is_team_yellow, 1)
    )

    #env_p3 = FootballEnv(wm, "127.0.0.1", 20011, 3, is_team_yellow, 1, True,)
    player_3 = Process(
        target=run_agent,
        # args=(halt_event, rollout_queue, env_p3, device, 3)
        args=(halt_event, rollout_queue, policy_state_queue, wm, device, 3, is_team_yellow, 1)
    )

    trainer = Process(
        target=train_model,
        args=(halt_event, rollout_queue, policy_state_queue, num_agents)
    )
    

    print("Starting agents...")
    goalie.start()
    player_1.start()
    player_2.start()
    player_3.start()
    trainer.start()


def run_referee(wm, halt_event):
    field = Field('B')
    referee = Referee(field, wm, halt_event)
    while True:
        referee.check()


if __name__ == '__main__':
    try:
        set_start_method('fork')
    except RuntimeError:
        pass
    vision_q = Queue()
    gc_q = Queue()  # GameController queue, not in use

    vision_wkr = Process(target=vision_worker, args=(vision_q, True, 10020,))

    wm_manager = WorldModelManager()
    wm_manager.start()
    wm = wm_manager.WorldModel()  # pylint: disable=no-member
    wmr = Process(target=wm_runner, args=(wm, vision_q, gc_q,))

    vision_wkr.start()
    wmr.start()

    halt_event = Event()

    referee_runner = Process(target=run_referee, args=(wm, halt_event,))

    referee_runner.start()

    mappo(wm, halt_event, is_team_yellow=True)
    mappo(wm, halt_event, is_team_yellow=False)

    referee_runner.join()
    vision_wkr.join()
    wmr.join()
