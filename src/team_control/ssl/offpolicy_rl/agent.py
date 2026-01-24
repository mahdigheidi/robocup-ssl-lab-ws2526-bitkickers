import torch

from team_control.ssl.rl.env import FootballEnv
from team_control.ssl.rl.actor_critic import Actor
from team_control.ssl.rl.encoder import Encoder


def load_actor():
    actor_model_path = "model_snapshots/actor/best_actor.pt"
    best_actor_dict = torch.load(actor_model_path)
    encoder = Encoder()
    actor = Actor(encoder)
    actor.load_state_dict(best_actor_dict['model_state_dict'])

    return actor


def agent_process(wm, division, sim_ip, cmd_listen_port, robot_id, is_team_yellow, role):
    training = False
    env = FootballEnv(
        wm, sim_ip, cmd_listen_port,
        division, robot_id,
        is_team_yellow, role, training,
    )
    actor = load_actor()
    actor.eval()
    robot, ball, teammates, opponents = env.reset()
    while True:
        action = actor(robot.unsqueeze(0), ball.unsqueeze(0), teammates.unsqueeze(0), opponents.unsqueeze(0))
        state, _, _ = env.step(action)
        robot, ball, teammates, opponents = state
