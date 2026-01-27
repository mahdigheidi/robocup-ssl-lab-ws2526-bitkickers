import numpy as np
from typing import Optional, Type, Any

from TeamControl.utils.state_machine import State, StateMachine
from TeamControl.network.robot_command import RobotCommand
from TeamControl.robot.context import RobotContext
from TeamControl.network.ssl_sockets import grSimSender
from TeamControl.ball.ball_idle_spawner import BallIdleSpawner
from TeamControl.obstacles.rules import avoid_defense_area_target


def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def norm2(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))

def unit(v: np.ndarray) -> np.ndarray:
    n = norm2(v)
    if n < 1e-6:
        return np.zeros_like(v)
    return v / n


SUPPORT_BEHIND_DIST = 1000.0
KICK_DISTANCE = 220.0
KICK_POWER = 8

MOVE_MAX_SPEED = 2000

# Goalie Configuration

GOALIE_POSITION_FRACTION = 0.2
GOALIE_SPEED_THRESHOLD = 0.5
GOALIE_GOAL_MARGIN = 100
GOALIE_BACKWARD_SHIFT_FACTOR = 1.0
GOALIE_MAX_ACCELERATION_BASE = 1500
GOALIE_MAX_ACCELERATION_DANGER = 6000


# States

class HoldPositionFaceBallState(State[RobotContext]):
    def __init__(self, home: np.ndarray):
        self.home = home

    def handle_transition(self, context):
        return None

    def execute(self, context, **kwargs):
        sender = kwargs["sender"]
        movement = kwargs["movement"]

        home_xy = np.array(self.home[:2], dtype=float)
        vx, vy = movement.go_to(context.my_state.position, home_xy, max_speed=MOVE_MAX_SPEED)
        w = movement.face_towards(context.my_state.position, context.ball.position)

        sender.send_command(
            RobotCommand(context.robot_id, vx, vy, w, kick=0, dribble=0)
        )


class StrikerBasicState(State[RobotContext]):
  
    "basic striker that goes to ball and kicks towards goal "
    def __init__(self, their_goal: np.ndarray):
        self.their_goal = np.array(their_goal[:2], dtype=float)

    def handle_transition(self, context):
        return None
 

    def execute(self, context, **kwargs):
        sender = kwargs["sender"]
        movement = kwargs["movement"]
        spawner: BallIdleSpawner = kwargs["spawner"]

        rx, ry, rtheta = context.my_state.position
        bx, by = context.ball.position

        # Clamp ball target to field bounds
      
        margin = 100
        bx = clamp(bx, spawner.field_x_min + margin, spawner.field_x_max - margin)
        by = clamp(by, spawner.field_y_min + margin, spawner.field_y_max - margin)

        bx, by = avoid_defense_area_target(
            bx,
            by,
            fallback_x=spawner.field_x_min if context.is_yellow else spawner.field_x_max,
        )

        dist_to_ball = np.hypot(bx - rx, by - ry)

       
        # Case 1: Far → chase ball
       
        if dist_to_ball > 110:
            vx, vy = movement.go_to(
                context.my_state.position,
                (bx, by),
                kicker_to_target=True,
                stop_threshold=5,
                max_speed=MOVE_MAX_SPEED,
            )
            w = movement.face_towards(
                robot_pos=context.my_state.position,
                target=(bx, by),
                max_angular_acceleration=3,
            )

            sender.send_command(
                RobotCommand(context.robot_id, vx, vy, w, kick=0, dribble=0)
            )
            return

        # Case 2: Align to goal

        goal_dir = np.arctan2(
            self.their_goal[1] - by,
            self.their_goal[0] - bx,
        )

        angle_error = np.arctan2(
            np.sin(goal_dir - rtheta),
            np.cos(goal_dir - rtheta),
        )

        if abs(np.degrees(angle_error)) > 4:
            omega = 4.0 * angle_error
            sender.send_command(
                RobotCommand(context.robot_id, 0, 0, omega, kick=0, dribble=1)
            )
            return

        # Case 3: STRIKE

        sender.send_command(
            RobotCommand(context.robot_id, 0, 0, 0, kick=KICK_POWER, dribble=0)
        )


class SupportBehindBallState(State[RobotContext]):
    "mid fielder function that goes behind the ball as of now "
    def __init__(self, our_goal: np.ndarray):
        self.our_goal = np.array(our_goal[:2], dtype=float)

    def handle_transition(self, context):
        return None

    def execute(self, context, **kwargs):
        sender = kwargs["sender"]
        movement = kwargs["movement"]
        spawner = kwargs.get("spawner")

        ball_xy = np.array(context.ball.position[:2], dtype=float)
        dir_to_goal = unit(self.our_goal - ball_xy)
        target = ball_xy + dir_to_goal * SUPPORT_BEHIND_DIST

        if spawner:
            margin = 1000
            target[0] = clamp(target[0], spawner.field_x_min + margin, spawner.field_x_max - margin)
            target[1] = clamp(target[1], spawner.field_y_min + margin, spawner.field_y_max - margin)

        vx, vy = movement.go_to(context.my_state.position, target, max_speed=MOVE_MAX_SPEED)
        w = movement.face_towards(context.my_state.position, ball_xy)

        sender.send_command(
            RobotCommand(context.robot_id, vx, vy, w, kick=0, dribble=0)
        )


class GoaliePositioningState(State[RobotContext]):
    "Goalie that positions itself based on ball trajectory."
    def handle_transition(self, context):
        return None

    def execute(self, context, **kwargs):
        sender: grSimSender = kwargs["sender"]
        movement = kwargs["movement"]
        spawner: BallIdleSpawner = kwargs["spawner"]

        robot_pos = context.my_state.position
        ball_pos = context.ball.position
        ball_vel = context.ball.velocity

        goal_sign = +1 if context.is_yellow else -1
        goal_x = goal_sign * spawner.goal_line_x

        ball_speed = np.linalg.norm(ball_vel)
        is_threat = False

        if ball_speed > GOALIE_SPEED_THRESHOLD and abs(ball_vel[0]) > 1e-6:
            slope = ball_vel[1] / ball_vel[0]
            iy = slope * (goal_x - ball_pos[0]) + ball_pos[1]

            if -spawner.goal_width / 2 - GOALIE_GOAL_MARGIN <= iy <= spawner.goal_width / 2 + GOALIE_GOAL_MARGIN:
                if goal_sign * ball_vel[0] > 0:
                    is_threat = True

        if is_threat:
            dx = robot_pos[0] - ball_pos[0]
            dy = robot_pos[1] - ball_pos[1]
            t = max(0, (dx * ball_vel[0] + dy * ball_vel[1]) / (np.dot(ball_vel, ball_vel)))
            target_x = ball_pos[0] + t * ball_vel[0]
            target_y = ball_pos[1] + t * ball_vel[1]
            max_acc = GOALIE_MAX_ACCELERATION_DANGER
        else:
            target_x = goal_x + GOALIE_POSITION_FRACTION * (ball_pos[0] - goal_x)
            target_y = GOALIE_POSITION_FRACTION * ball_pos[1]
            max_acc = GOALIE_MAX_ACCELERATION_BASE

        vx, vy = movement.go_to(robot_pos, (target_x, target_y), max_acceleration=max_acc)
        w = movement.face_towards(robot_pos, ball_pos)

        sender.send_command(
            RobotCommand(context.robot_id, vx, vy, w, kick=5, dribble=0)
        )


# StateMachines
class HoldPositionFaceBallStateMachine(StateMachine):
    def __init__(self, home):
        super().__init__(HoldPositionFaceBallState(home))


class StrikerBasicStateMachine(StateMachine):
    def __init__(self, their_goal):
        super().__init__(StrikerBasicState(their_goal))


class SupportBehindBallStateMachine(StateMachine):
    def __init__(self, our_goal):
        super().__init__(SupportBehindBallState(our_goal))


class GoalieStateMachine(StateMachine):
    def __init__(self):
        super().__init__(GoaliePositioningState())
