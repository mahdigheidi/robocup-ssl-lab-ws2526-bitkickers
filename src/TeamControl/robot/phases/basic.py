import math
from venv import logger
from TeamControl.SSL.grSim.interceptor_process import avoid_collision, block_pass_point, is_safe_from_robots, time_to_reach
from TeamControl.network import sender
from TeamControl.robot import context
from TeamControl.robot.phases.ball_handling import calculate_receiver_position, is_path_blocked
import numpy as np
from typing import Optional, Type, Any

from TeamControl.utils.state_machine import State, StateMachine
from TeamControl.network.robot_command import RobotCommand
from TeamControl.robot.context import RobotContext
from TeamControl.network.ssl_sockets import grSimSender
from TeamControl.ball.ball_idle_spawner import BallIdleSpawner
from TeamControl.obstacles.rules import avoid_defense_area_target
from TeamControl.obstacles.helpers import RoleArea

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def norm2(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))

def unit(v: np.ndarray) -> np.ndarray:
    n = norm2(v)
    if n < 1e-6:
        return np.zeros_like(v)
    return v / n

STRIKER_ID = 3

BALL_SPEED_THRESHOLD = 500.0
DT = 0.016
STEPS = 30
SAFE_DISTANCE = 600.0
MAX_SPEED = 3000.0

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
def shadow_ball_target(
    context: RobotContext,
    clamp_fn,
    margin: float = 300.0,
) -> np.ndarray:
    bx, by = context.ball.position[:2]
    rx, ry = context.my_state.position[:2]

    ball = np.array([bx, by])
    robot = np.array([rx, ry])

    anchor = clamp_fn(context, bx, by)

    to_ball = ball - anchor
    norm = np.linalg.norm(to_ball)
    if norm < 1e-6:
        return anchor

    dir_to_ball = to_ball / norm
    perp = np.array([-dir_to_ball[1], dir_to_ball[0]], dtype=float)

    left = anchor + perp * margin
    right = anchor - perp * margin

    target = left if np.linalg.norm(left - robot) < np.linalg.norm(right - robot) else right

    return clamp_fn(context, float(target[0]), float(target[1]))


class ShadowBallState(State[RobotContext]):
    def __init__(self, role_area: RoleArea, margin: float = 300.0):
        self.role_area = role_area
        self.margin = margin

    def handle_transition(self, context: RobotContext):
        # Role controls transitions (SHADOW → CHASE etc.)
        return None    

    def execute(self, context: RobotContext, **kwargs):
        sender = kwargs["sender"]
        movement = kwargs["movement"]

        target = shadow_ball_target(
            context,
            clamp_fn=lambda ctx, x, y: self.role_area.project_to_legal_area(ctx, x, y),
            margin=self.margin,
        )

        vx, vy = movement.go_to(
            context.my_state.position,
            target,
            max_speed=1500,
        )
        w = movement.face_towards(context.my_state.position, context.ball.position)

        sender.send_command(
            RobotCommand(
                context.robot_id,
                vx, vy, w,
                kick=0,
                dribble=0,
            )
        )


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



class SupportBehindBallState(State[RobotContext]):
    "mid fielder function that goes behind the ball as of now "
    def __init__(self, yellow_goal: np.ndarray):
        self.yellow_goal = np.array(yellow_goal[:2], dtype=float)

    def handle_transition(self, context):
        return None

    def execute(self, context, **kwargs):
        sender = kwargs["sender"]
        movement = kwargs["movement"]
        spawner = kwargs.get("spawner")

        ball_xy = np.array(context.ball.position[:2], dtype=float)
        dir_to_goal = unit(self.yellow_goal - ball_xy)
        target = ball_xy + dir_to_goal * SUPPORT_BEHIND_DIST

        if spawner:
            margin = 100
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
            RobotCommand(context.robot_id, vx, vy, w, kick=0, dribble=0)
        )
        if context.has_possession and not context.teammates:
            sender.send_command(
                RobotCommand(context.robot_id, 0, 0, 0, kick=7, dribble=0)
        )


class BlockPassingLaneState(State[RobotContext]):
    """Apply pressure by standing near nearest opponent and facing the ball."""

    PRESSURE_DISTANCE = 20.0  # mm offset from opponent toward ball

    def handle_transition(self, context: RobotContext):
        ball_speed = np.linalg.norm(context.ball.velocity)

        if context.has_possession:
            return None  # Role handles exit

        # Switch to interception if ball is fast
        if ball_speed > BALL_SPEED_THRESHOLD:
            return InterceptFastBallState

        return None

    def execute(self, context: RobotContext, **kwargs):
        sender: grSimSender = kwargs["sender"]
        movement = kwargs["movement"]

        if not context.opponents:
            return

        rx, ry, rtheta = context.my_state.position
        ball_pos = np.array(context.ball.position[:2], dtype=float)

        # 1. Find nearest opponent
        nearest_opponent = min(
            context.opponents.values(),
            key=lambda op: np.linalg.norm(op.position[:2] - np.array([rx, ry]))
        )

        opp_pos = np.array(nearest_opponent.position[:2], dtype=float)

        # 2. Stand between opponent and ball
        opp_to_ball = ball_pos - opp_pos
        dist = np.linalg.norm(opp_to_ball)

        if dist > 1e-6:
            dir_to_ball = opp_to_ball / dist
        else:
            dir_to_ball = np.zeros(2)

        target = opp_pos + dir_to_ball * self.PRESSURE_DISTANCE

        # 3. Move with low speed, face ball
        vx, vy = movement.go_to(
            context.my_state.position,
            target,
            max_speed=MAX_SPEED , 
        )

        w = movement.face_towards(
            context.my_state.position,
            ball_pos,
        )

        sender.send_command(
            RobotCommand(
                context.robot_id,
                vx, vy, w,
                kick=0,
                dribble=0,
            )
        )



class InterceptFastBallState(State[RobotContext]):
    """Intercept a fast-moving ball using future trajectory prediction."""
    def __init__(self, role_area: RoleArea| None = None):
        self.role_area = role_area

    def handle_transition(self, context: RobotContext):
        ball_speed = np.linalg.norm(context.ball.velocity)

        if context.has_possession:

            return None  # Role handles exit

        # Go back to blocking if ball slows down
        #if ball_speed < 500:
            #return BlockPassingLaneState

        return None

    def execute(self, context: RobotContext, **kwargs):
        sender: grSimSender = kwargs["sender"]
        movement = kwargs["movement"]

        rx, ry, rtheta = context.my_state.position
        ball_x, ball_y = context.ball.position
        ball_vx, ball_vy = context.ball.velocity

        opponents = [op.position for op in context.opponents.values()]

        best_target = None
        best_diff = float("inf")

        for k in range(1, STEPS + 1):
            bx_f = ball_x + ball_vx * k * DT
            by_f = ball_y + ball_vy * k * DT

            if self.role_area is not None:
                if not self.role_area.is_point_legal(context, bx_f, by_f):
                    continue

            if not is_safe_from_robots(bx_f, by_f, opponents):
                continue

            dist_mm = np.hypot(bx_f - rx, by_f - ry)
            t_robot = time_to_reach(dist_mm)
            t_ball = k * 0.016

            diff = abs(t_robot - t_ball)
            if diff < best_diff:
                best_diff = diff
                best_target = (bx_f, by_f)

        if best_target is None:
            return

        tx, ty = best_target
       

        vx, vy = movement.go_to(
            context.my_state.position,
            (tx, ty),
            max_speed=3000,
        )
        w = movement.face_towards(context.my_state.position, context.ball.position)

        sender.send_command(
            RobotCommand(context.robot_id, vx, vy, w, kick=0, dribble=1)
        )


       
class KickToGoalState(State[RobotContext]):
    """
    Striker-only shooting state.
    - Samples goal mouth
    - Clears lane naturally
    - Aligns
    - Kicks
    """

    GOAL_SAMPLE_Y = np.linspace(-500, 500, 7)
    ALIGN_THRESHOLD_DEG = 4.0
    REPOSITION_SPEED = 1500

    def handle_transition(self, context: RobotContext):
        # Role controls exit
        if not context.has_possession:
            return None
        return None

    def execute(self, context: RobotContext, **kwargs):
        sender: grSimSender = kwargs["sender"]
        movement = kwargs["movement"]
        spawner: BallIdleSpawner = kwargs["spawner"]

        rx, ry, rtheta = context.my_state.position
        bx, by = context.ball.position

        # --------------------------------------------
        # 1. Pick goal x
        # --------------------------------------------
        goal_x = -4500.0 if context.is_yellow else 4500.0
        ball_xy = np.array([bx, by])

        opponent = list(context.opponents.values())[0] if context.opponents else None

        clear_target = None
        reposition_target = None

        # --------------------------------------------
        # 2. Search for open goal lane
        # --------------------------------------------
        for gy in self.GOAL_SAMPLE_Y:
            goal_target = np.array([goal_x, gy])

            if opponent is None:
                clear_target = goal_target
                break

            blocked = is_path_blocked(
                passer_pos=ball_xy,
                receiver_pos=goal_target,
                opponent_pos=opponent.position,
            )

            if not blocked:
                clear_target = goal_target
                break

            # Save last reposition attempt
            reposition_target = calculate_receiver_position(
                receiver_pos=ball_xy,
                passer_pos=goal_target,
                opponent_state=opponent,
                field_x_min=spawner.field_x_min,
                field_x_max=spawner.field_x_max,
                field_y_min=spawner.field_y_min,
                field_y_max=spawner.field_y_max,
            )

        # --------------------------------------------
        # 3. No clear lane → reposition naturally
        # --------------------------------------------
        if clear_target is None and reposition_target is not None:
            vx, vy = movement.go_to(
                context.my_state.position,
                reposition_target,
                max_speed=self.REPOSITION_SPEED,
            )
            w = movement.face_towards(context.my_state.position, reposition_target)

            sender.send_command(
                RobotCommand(
                    context.robot_id,
                    vx, vy, w,
                    kick=0,
                    dribble=1,
                )
            )
            return

        # --------------------------------------------
        # 4. Align to goal
        # --------------------------------------------
        dx = clear_target[0] - bx
        dy = clear_target[1] - by
        goal_angle = math.atan2(dy, dx)

        angle_error = (goal_angle - rtheta + math.pi) % (2 * math.pi) - math.pi
        angle_error_deg = abs(math.degrees(angle_error))

        if angle_error_deg > self.ALIGN_THRESHOLD_DEG:
            w = 4.0 * angle_error
            sender.send_command(
                RobotCommand(
                    context.robot_id,
                    vx=0,
                    vy=0,
                    w=w,
                    kick=0,
                    dribble=1,
                )
            )
            return

        # --------------------------------------------
        # 5. KICK
        # --------------------------------------------
        sender.send_command(
            RobotCommand(
                context.robot_id,
                vx=0,
                vy=0,
                w=0,
                kick=KICK_POWER,
                dribble=0,
            )
        )

class KickToGoalStateMachine(StateMachine[RobotContext]):
    def __init__(self):
        super().__init__(initial_state=KickToGoalState())

class InterceptionStateMachine(StateMachine[RobotContext]):
    """Intercept fast balls or goalie possession."""

    def __init__(self, role_area: RoleArea):
        intercept = InterceptFastBallState(role_area)
        super().__init__(initial_state=intercept)


class HoldPositionFaceBallStateMachine(StateMachine):
    def __init__(self, home):
        super().__init__(HoldPositionFaceBallState(home))


class SupportBehindBallStateMachine(StateMachine):
    def __init__(self, yellow_goal):
        super().__init__(SupportBehindBallState(yellow_goal))   

class GoalieStateMachine(StateMachine):
    def __init__(self):
        super().__init__(GoaliePositioningState())

class ShadowBallStateMachine(StateMachine):
    def __init__(self, role_area: RoleArea, margin: float = 300.0):
        super().__init__(ShadowBallState(role_area, margin=margin))