import math
import random
import traceback
from typing import List

import torch
from tensordict import TensorDict

from team_control.network.robot_command import RobotCommand
from team_control.network.sender import GrSimSender
from team_control.network.send_ball_grsim import BallSender
from team_control.world.model import WorldModel
from team_control.ssl.rl.field import Field
from team_control.ssl.vision import Ball, Robot


class GrSimEnv:
    def __init__(
        self,
        wm: WorldModel,
        sim_ip: str,
        cmd_listen_port: int,
        training: bool = True,
    ):
        """
        Args:
            wm: WorldModel instance for robot and ball states
            sim_ip: IP address of grSim
            cmd_listen_port: port to send commands to grSim
            is_team_yellow: whether our team is yellow (True) or blue (False)
            training: whether environment is in training mode (affects reward)
        """
        super().__init__()

        self.wm: WorldModel = wm
        self.sim_ip = sim_ip
        self.cmd_listen_port = cmd_listen_port
        self.training = training

        self.sender_yellow = GrSimSender(ip=self.sim_ip, port=self.cmd_listen_port, is_yellow=True)
        self.sender_blue = GrSimSender(ip=self.sim_ip, port=self.cmd_listen_port, is_yellow=False)
        self.ball_sender = BallSender()

        self.field = Field("B")
        self.field_x = self.field.x_max
        self.field_y = self.field.y_max

        self.version = 0

        # These will be updated on each world model update
        self.ball: Ball = None
        self.robots_yellow: List[Robot] = []
        self.robots_blue: List[Robot] = []

        # Number of agents = number of robots on both teams
        self.num_agents = 4 * 2  # 4 robots per team * 2 teams

        # Observation dimension calculation:
        # self: 4 (x,y,sin(o),cos(o))
        # ball relative pos: 2 (dx, dy)
        # yellow relative pos: (3 + 1) * 2 = 8
        # blue relative pos: (3 + 1) * 2 = 8
        self.obs_dim = 22

        # State dimension for centralized critic:
        # ball: 2 (x,y)
        # all robots: 8 (num_agents) * 4 (x,y,sin(o),cos(o)) = 32
        self.state_dim = 34

        # Action dimension per agent: [vx, vy, w, kick, dribble]
        # We'll define action as continuous bounded for vx, vy, w; kick and dribble as continuous [0,1]
        self.action_dim = 5

    def reset(self) -> TensorDict:
        """
        Resets environment spawning the ball in a random point inside the center circle.
        Waits for a new world model update.
        Returns observations, state.
        """
        r = random.uniform(0, self.field.radius)
        theta = random.uniform(0, 2*math.pi)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        self.ball_sender.send_ball_to_grsim(x, y, 0, 0)
        while True:
            if self._wait_for_wm_update():
                break

    def step(self, actions) -> TensorDict:
        for i in range(len(actions)):
            id = i if i < 3 else i - 3
            is_yellow = True if i < 3 else False
            sender = self.sender_yellow if is_yellow else self.sender_blue

            action = actions[i]
            vx, vy, w, kick, dribble = action[0]

            # TODO: clipping the actions in the range
            # vx is in range [-5, 5]
            # vy is in range [-5, 5]
            # w is in range [-10, 10]
            # kick is in range [0, 10]
            # dribble is in range [0, 1]
            vx = max(-5.0, min(5.0, vx))
            vy = max(-5.0, min(5.0, vy))
            w = max(-10.0, min(10.0, w))
            kick = max(0.0, min(10.0, kick))
            dribble = max(0.0, min(1.0, dribble))
            dribble = 1 if dribble > 0.5 else 0

            cmd = RobotCommand(
                robot_id=id,
                vx=vx,
                vy=vy,
                w=w,
                kick=kick,
                dribble=dribble,
            )
            sender.send_command(cmd)

        # Wait for next world model update
        while True:
            if self._wait_for_wm_update():
                break

        game_state = self._get_global_state()
        agent_observations = self._get_agent_observation()

        # Compute shared team rewards
        reward_yellow, done = self._compute_team_rewards()

        return game_state, agent_observations, reward_yellow, done

    def _wait_for_wm_update(self) -> bool:
        """
        Checks if new world model data is available.
        Updates internal ball and robot states.
        Returns True if updated, False otherwise.
        """
        version = self.wm.get_version()
        if version == self.version:
            return False
        self.version = version
        try:
            self.robots_yellow = self.wm.get_our_robots(us=True)
            self.robots_blue = self.wm.get_our_robots(us=False)
            self.ball = self.wm.get_latest_frame().ball

        except Exception as e:
            print("WM update error:", e)
            traceback.print_exc()
            return False

        if self.ball is None or len(self.robots_yellow) == 0 or len(self.robots_blue) == 0:
            raise Exception("World model returned invalid ball or robot data")

        return True

    def _get_global_state(self) -> torch.Tensor:
        """
        Returns global absolute state vector for centralized critic.

        Structure:
        - ball: (x, y, vx, vy) normalized by field size and max velocity (assumed)
        - all robots (yellow then blue): (x, y, sin(theta), cos(theta), vx, vy)

        Positions normalized by field dimensions.
        """

        # Ball position and velocity normalized
        ball_x, ball_y = self.ball.position
        ball_state = torch.tensor(
            [
                ball_x / self.field_x,
                ball_y / self.field_y,
            ],
            dtype=torch.float32,
        )

        def robot_state(robot: Robot) -> torch.Tensor:
            x, y, o = robot.position
            return torch.tensor(
                [
                    x / self.field_x,
                    y / self.field_y,
                    math.sin(o),
                    math.cos(o),
                ],
                dtype=torch.float32,
            )

        robots_yellow_states = []
        for i in range(4):
            robots_yellow_states.append(robot_state(self.robots_yellow[i]))
        robots_yellow_states = torch.cat([x for x in robots_yellow_states])

        robots_blue_states = []
        for i in range(4):
            robots_blue_states.append(robot_state(self.robots_blue[i]))
        robots_blue_states = torch.cat([x for x in robots_blue_states])

        state = torch.cat([ball_state, robots_yellow_states, robots_blue_states], dim=0)
        return state

    def _get_agent_observation(self) -> torch.Tensor:
        """
        Constructs relative observation vector for a single agent.

        Observation includes:
        - self: (x, y, sin(theta), cos(theta))
        - ball: relative (dx, dy)
        - teammates: relative (dx, dy) for all teammates (excluding self), zero-padded to MAX_TEAMMATES
        - opponents: relative (dx, dy) for all opponents, zero-padded to MAX_OPPONENTS

        Positions normalized by field dimensions.
        """
        observations = []  # array of size 6  -  3 yellow and 3 blue
        bx, by = self.ball.position

        for i in range(1, 4):
            x, y, o = self.robots_yellow[i].position
            self_state = torch.tensor(
                [x / self.field_x, y / self.field_y, math.sin(o), math.cos(o)], dtype=torch.float32
            )
            ball_rel = torch.tensor([(bx - x) / self.field_x, (by - y) / self.field_y], dtype=torch.float32)
            yellow_rel = []
            for j in range(4):
                mx, my, _ = self.robots_yellow[j].position
                dx = (mx - x) / self.field_x
                dy = (my - y) / self.field_y
                yellow_rel.append(torch.tensor([dx, dy], dtype=torch.float32))
            yellow_rel = torch.cat([x for x in yellow_rel])
            blue_rel = []
            for j in range(4):
                mx, my, _ = self.robots_blue[j].position
                dx = (mx - x) / self.field_x
                dy = (my - y) / self.field_y
                blue_rel.append(torch.tensor([dx, dy], dtype=torch.float32))
            blue_rel = torch.cat([x for x in blue_rel])
            observation = torch.cat([self_state, ball_rel, yellow_rel, blue_rel])
            observations.append(observation)

        for i in range(1, 4):
            x, y, o = self.robots_blue[i].position
            self_state = torch.tensor(
                [x / self.field_x, y / self.field_y, math.sin(o), math.cos(o)], dtype=torch.float32
            )
            ball_rel = torch.tensor([(bx - x) / self.field_x, (by - y) / self.field_y], dtype=torch.float32)
            yellow_rel = []
            for j in range(4):
                mx, my, _ = self.robots_yellow[j].position
                dx = (mx - x) / self.field_x
                dy = (my - y) / self.field_y
                yellow_rel.append(torch.tensor([dx, dy], dtype=torch.float32))
            yellow_rel = torch.cat([x for x in yellow_rel])
            blue_rel = []
            for j in range(4):
                mx, my, _ = self.robots_blue[j].position
                dx = (mx - x) / self.field_x
                dy = (my - y) / self.field_y
                blue_rel.append(torch.tensor([dx, dy], dtype=torch.float32))
            blue_rel = torch.cat([x for x in blue_rel])
            observation = torch.cat([self_state, ball_rel, yellow_rel, blue_rel])
            observations.append(observation)

        return observations

    def _compute_team_rewards(self):
        """
        Compute shared team rewards for yellow and blue teams.

        This function replaces per-agent reward logic.
        Reward is computed at the team level, encouraging cooperation.

        For simplicity, this example returns zero rewards.
        Replace this method with your custom team-level reward logic.

        Returns:
            (reward_yellow: float, reward_blue: float)
        """
        bx, by = self.ball.position
        # Goal is scored by yellow team
        if bx <= self.field.x_min and -self.field.goal_width / 2 <= by <= self.field.goal_width / 2:
            return 1, True
        # Goal is scored by blue team
        if bx >= self.field.x_max and -self.field.goal_width / 2 <= by <= self.field.goal_width / 2:
            return -1, True

        # if ball is outside field the game is done, nobody recieves nothing, game is done
        if bx > self.field.x_max or bx < self.field.x_min or by > self.field.y_max or by < self.field.y_min:
            return 0, True

        # if an agent of the yellow team is inside their own or other teams defense area -0.33 but its just a foul and game continues
        for i in range(1, 4):
            rx, ry, _ = self.robots_yellow[i].position
            if -self.field.penalty_width / 2 <= ry <= self.field.penalty_width / 2 and (rx >= self.field.x_max-self.field.penalty_depth or rx <= self.field.x_min+self.field.penalty_depth):
                return -0.33, False

        # if the yellow team has the ball in possession they are rewarded 0.05 and game continues
        for i in range(4):
            rx, ry, _ = self.robots_yellow[i].position
            if math.hypot(rx-bx, ry-by) <= 110:
                return 0.05, False
        # if the blue team has the ball in possession they are punished -0.05 and game continues
        for i in range(4):
            rx, ry, _ = self.robots_blue[i].position
            if math.hypot(rx-bx, ry-by) <= 110:
                return -0.05, False

        return -0.01, False
