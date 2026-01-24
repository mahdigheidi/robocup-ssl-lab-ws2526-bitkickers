import math
import random
import traceback
from typing import List

import torch

from team_control.network.robot_command import RobotCommand
from team_control.network.sender import GrSimSender
from team_control.network.send_ball_grsim import BallSender
from team_control.robot.spawn_robot import send_robot_to_grsim
from team_control.world.model import WorldModel
from team_control.ssl.rl.field import Field
from team_control.ssl.rl.reward import Reward
from team_control.ssl.vision import Ball, Robot


# CONFIG THE FOOTBALL ENV FOR PLAYERS ON THE LEFT SIDE

class FootballEnv:
    def __init__(self, wm, sim_ip, cmd_listen_port, robot_id, is_team_yellow, role, training):
        """
        wm: WorldModel instance (for getting robot/ball states)
        spawner: Object containing goal_line_x or other default positions
        sim_ip: IP of grSim
        cmd_listen_port: port to send commands
        division: information about the physics of the field
        robot_id: an integer representing the ID of the robot taking the command
        is_team_yellow: a boolean indicating if we are playing yellow
        role: the role of the player, can be either 0(goalie) or 1(field player)
        """
        # GrSim
        self.wm: WorldModel = wm
        self.version = 0
        self.sim_ip = sim_ip
        self.cmd_listen_port = cmd_listen_port
        self.ball_sender = BallSender()
        self.sender = GrSimSender(ip=self.sim_ip, port=self.cmd_listen_port, is_yellow=is_team_yellow)

        # Curriculum
        # self.curriculum = curriculum

        # Field Physics
        self.field = Field('B')
        self.field_x = self.field.x_max
        self.field_y = self.field.y_max

        self.robot_id = robot_id
        self.is_team_yellow = is_team_yellow
        self.role = role       # 0 is Goalie, 1 is Field player

        self.robot: Robot = None
        self.ball: Ball = None
        self.teammates: List[Robot] = None
        self.opponents: List[Robot] = None

        # Actions scaling
        self.move_scale = 5.0
        self.rotate_scale = 5.0
        self.kick_scale = 10.0

        self.training = training
        # Reward function
        self.reward = Reward(role)

        self.reset()

    def reset(self) -> torch.Tensor:
        """
        Spawns ball and robot randomly, updates robot_id and team
        """

        # Spawn ball
        # radius = self.field.radius
        # x_ball = random.randint(-radius, radius)
        # y_ball = random.randint(-radius, radius)
        # self.ball_sender.send_ball_to_grsim(x_ball, y_ball, 0, 0)

        # Spawn robot
        # if self.role == 1:
        #     x = random.randint(-(self.field_x-self.field.penalty_depth), self.field_x-self.field.penalty_depth)
        #     y = random.randint(-self.field_y, self.field_y)
        #     orientation = random.uniform(0, 360)
        # else:
        #     a, b = self.field_x-self.field.penalty_depth, self.field_x
        #     if not self.is_team_yellow:
        #         a, b = -b, -a
        #     x = random.randint(a, b)
        #     y = random.randint(-self.field.penalty_width//2, self.field.penalty_width//2)
        #     orientation = random.randint(90, 270)
        # send_robot_to_grsim(self.is_team_yellow, self.robot_id, x, y, orientation)

        # Wait for wm update
        while True:
            print("Waiting for WM update...")
            if self._wait_for_wm_update():
                break

        # Return normalized state
        return self.get_normalized_state()

    # ----------------------
    # Step function
    # ----------------------
    def step(self, action):
        """
        action = [vx, vy, w, kick_power, dribble]
        Sends the action to the simulator and waits for next world model update
        """
        # print(action)
        vx, vy, w, kick, dribble = action[0]
        # vx_mask, vy_mask, w_mask, kick_mask, dribble_mask = self.curriculum.action_mask

        cmd = RobotCommand(
            robot_id=self.robot_id,
            vx=vx,
            vy=vy,
            w=w,
            kick=kick,
            dribble=dribble,
        )

        while True:
            self.sender.send_command(cmd)
            if self._wait_for_wm_update():
                break

        if self.training:
            reward, done = self.reward.calculate_reward(
                self.field, self.is_team_yellow, self.robot, self.ball, self.teammates, self.opponents)
        else:
            reward, done = 0, False

        state = self.get_normalized_state()

        return state, reward, done

    def get_normalized_state(self):
        robot_state = _get_robot_state(self.field, self.robot, self.role, self.is_team_yellow)
        ball_state = _get_ball_state(self.field, self.ball)

        num_teammates = 0
        available_teammates = []
        for teammate in self.teammates:
            if isinstance(teammate, Robot):
                num_teammates += 1
                available_teammates.append(teammate)
        teammates_state = torch.zeros(max(num_teammates, 1), 6)
        # print("Len teammates is: ", num_teammates)
        for i in range(len(available_teammates)):
            teammates_state[i] = _get_robot_state(
                self.field, available_teammates[i], get_role(available_teammates[i].position), self.is_team_yellow,
            )

        num_opponents = 0
        available_opponents = []
        for opponent in self.opponents:
            if isinstance(opponent, Robot):
                num_opponents += 1
                available_opponents.append(opponent)
        opponents_state = torch.zeros(max(num_opponents, 1), 6)
        # print("Len opponents is: ", num_opponents)
        for i in range(len(available_opponents)):
            opponents_state[i] = _get_robot_state(
                self.field, available_opponents[i], get_role(available_opponents[i].position), not self.is_team_yellow,
            )
        return robot_state.unsqueeze(0), ball_state.unsqueeze(0), teammates_state, opponents_state

    # ----------------------
    # Internal helpers
    # ----------------------
    def _wait_for_wm_update(self):
        version = self.wm.get_version()
        if version == self.version:
            return False
        self.version = version
        try:
            robot_obj, ball_obj, teammates_obj, opponents_obj = self.wm.get_robot_context(
                self.robot_id, self.is_team_yellow)
            self.robot = robot_obj
            # print(self.robot)
            # print("-"*80)
            self.ball = ball_obj
            self.teammates = []
            for teammate in teammates_obj:
                if isinstance(teammate, Robot):
                    self.teammates.append(teammate)
            self.opponents = []
            for opponent in opponents_obj:
                if isinstance(opponent, Robot):
                    self.opponents.append(opponent)
        except Exception as e:
            print("WM update error:", e)
            traceback.print_exc()
        if self.robot is None or self.ball is None:
            raise Exception("World model has returned a none ball or robot")
        return True


def _get_robot_state(field, robot, role, right_side_playing):
    x, y, o = robot.position
    return torch.tensor([
        x / field.x_max, y / field.y_max,
        math.sin(o), math.cos(o),
        role,
        1 if right_side_playing else 0,
    ], dtype=torch.float32)


def _get_ball_state(field, ball):
    x, y = ball.position
    return torch.tensor([x / field.x_max, y / field.y_max], dtype=torch.float32)


def get_my_normalized_state(robot, ball, teammates, opponents, is_team_yellow):
    field = Field('B')
    print(robot, ball, teammates, opponents)
    print("-"*100)
    robot_state = _get_robot_state(field, robot, get_role(robot.position), is_team_yellow)
    ball_state = _get_ball_state(field, ball)
    num_teammates = 0
    available_teammates = []
    for teammate in teammates:
        if isinstance(teammate, Robot):
            num_teammates += 1
            available_teammates.append(teammate)
    teammates_state = torch.zeros(max(num_teammates, 1), 6)
    # print("Len teammates is: ", num_teammates)
    for i in range(len(available_teammates)):
        teammates_state[i] = _get_robot_state(
            field, available_teammates[i], get_role(available_teammates[i].position), is_team_yellow,
        )

    num_opponents = 0
    available_opponents = []
    for opponent in opponents:
        if isinstance(opponent, Robot):
            num_opponents += 1
            available_opponents.append(opponent)
    opponents_state = torch.zeros(max(num_opponents, 1), 6)
    # print("Len opponents is: ", num_opponents)
    for i in range(len(available_opponents)):
        opponents_state[i] = _get_robot_state(
            field, available_opponents[i], get_role(available_opponents[i].position), not is_team_yellow,
        )

    return robot_state.unsqueeze(0), ball_state.unsqueeze(0), teammates_state, opponents_state


def get_role(position):
    x, y, _ = position
    if -1000 <= y <= 1000 and (x >= 3300 or x <= -3300):
        return 0
    return 1
