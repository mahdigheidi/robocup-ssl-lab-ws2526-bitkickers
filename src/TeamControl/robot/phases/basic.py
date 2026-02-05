import math
import time
from venv import logger
from TeamControl.SSL.grSim.interceptor_process import avoid_collision, block_pass_point, is_safe_from_robots, time_to_reach, safe_target
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
from TeamControl.obstacles.helpers import RoleArea, avoid_collision_robust

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
STEPS = 300
SAFE_DISTANCE = 600.0
MAX_SPEED = 3000.0

KICK_DISTANCE = 220.0
KICK_POWER = 8

MOVE_MAX_SPEED = 2000

# Goalie Configuration

GOALIE_POSITION_FRACTION = 0.1
GOALIE_SPEED_THRESHOLD = 0.5
GOALIE_GOAL_MARGIN = 100
GOALIE_BACKWARD_SHIFT_FACTOR = 1.0
GOALIE_MAX_ACCELERATION_BASE = 3000
GOALIE_MAX_ACCELERATION_DANGER = 6000


def pick_open_goal(context: RobotContext, spawner: BallIdleSpawner, force_center: bool = False) -> np.ndarray:
    """Find an unblocked point on the opponent's goal line."""
    goal_x = -spawner.goal_line_x if context.is_yellow else spawner.goal_line_x
    
    if force_center:
        return np.array([goal_x, 0.0])
    
    # Range of y in goal area: -400 to 400
    y_points = np.linspace(-400, 400, 59) # 9 points: -400, -300, ..., 400
    
    ball_xy = context.ball.position[:2]
    opponents = list(context.opponents.values())
    
    # Priority: midpoint first
    mid_idx = len(y_points) // 2
    indices = [mid_idx]
    for i in range(1, mid_idx + 1):
        indices.append(mid_idx + i)
        indices.append(mid_idx - i)
        
    for idx in indices:
        y = y_points[idx]
        target = np.array([goal_x, y])
        
        blocked = False
        for op in opponents:
            if is_path_blocked(ball_xy, target, op.position):
                blocked = True
                break
        
        if not blocked:
            return target
            
    # Fallback to midpoint if all blocked
    return np.array([goal_x, 0.0])


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

        # Prepare obstacles for movement
        static_obs = [tuple(o.position[:2]) for o in context.opponents.values()] + \
                     [tuple(t.position[:2]) for t in context.teammates.values()]

        vx, vy = movement.go_to(
            context.my_state.position,
            target,
            max_speed=1500,
            static_obstacles=static_obs,
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





class GoaliePositioningState(State[RobotContext]):
    "Goalie that positions itself based on ball trajectory."
    def __init__(self, role_area: Optional[RoleArea] = None):
        self.role_area = role_area

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

        # Avoid collision with other robots
        obstacles = list(context.opponents.values()) + list(context.teammates.values())
        static_obs = [tuple(o.position[:2]) for o in obstacles]
        target_x, target_y = avoid_collision_robust(robot_pos[0], robot_pos[1], target_x, target_y, obstacles)

        # Project to legal area if provided
        if self.role_area is not None:
            tx, ty = self.role_area.project_to_legal_area(context, target_x, target_y, is_goalie=True)
            target_x, target_y = tx, ty

        vx, vy = movement.go_to(robot_pos, (target_x, target_y), max_acceleration=max_acc, static_obstacles=static_obs)
        w = movement.face_towards(robot_pos, ball_pos)
        sender.send_command(
            RobotCommand(context.robot_id, vx, vy, w, kick=0, dribble=0)
        )
        if context.has_possession and not context.teammates:
            sender.send_command(
                RobotCommand(context.robot_id, 0, 0, 0, kick=7, dribble=0)
        )


def is_opponent_possession(context: RobotContext) -> bool:
    """Check if any opponent robot has possession of the ball."""
    ball_pos = context.ball.position[:2]
    for op in context.opponents.values():
        if np.linalg.norm(op.position[:2] - ball_pos) < 120.0:  # POSSESSION_THRESHOLD
            return True
    return False

class BlockPassingLaneState(State[RobotContext]):
    """Apply pressure by standing near ball/opponent and facing the ball."""

    BLOCK_DISTANCE = 500.0  # mm distance from ball to block

    def __init__(self, role_area: Optional[RoleArea] = None):
        self.role_area = role_area

    def handle_transition(self, context: RobotContext):
        ball_speed = np.linalg.norm(context.ball.velocity)

        if context.has_possession:
            return None  # Role handles exit

        # Switch to interception if ball is fast and no one has it
        if ball_speed > 500.0 and not is_opponent_possession(context):
            return InterceptFastBallState

        return None

    def execute(self, context: RobotContext, **kwargs):
        sender: grSimSender = kwargs["sender"]
        movement = kwargs["movement"]

        rx, ry, rtheta = context.my_state.position
        ball_pos = np.array(context.ball.position[:2], dtype=float)
        
        # Target: Position 500mm away from the ball
        to_robot = np.array([rx, ry]) - ball_pos
        dist = np.linalg.norm(to_robot)
        
        if dist > 1e-6:
            dir_from_ball = to_robot / dist
        else:
            goal_sign = -1 if context.is_yellow else 1
            dir_from_ball = np.array([goal_sign, 0.0])

        target = ball_pos + dir_from_ball * self.BLOCK_DISTANCE
        tx, ty = target[0], target[1]

        # Avoid collision with other robots
        obstacles = list(context.opponents.values()) + list(context.teammates.values())
        static_obs = [tuple(o.position[:2]) for o in obstacles]
        tx, ty = avoid_collision_robust(rx, ry, tx, ty, obstacles)

        # Apply RoleArea constraints
        if self.role_area is not None:
            tx, ty = self.role_area.project_to_legal_area(context, tx, ty)

        vx, vy = movement.go_to(
            context.my_state.position,
            (tx, ty),
            max_speed=MAX_SPEED, 
            static_obstacles=static_obs,
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

        # Go back to blocking if ball slows down or opponent gets it
        if ball_speed < 400.0 or is_opponent_possession(context):
            return BlockPassingLaneState

        return None

    def execute(self, context: RobotContext, **kwargs):
        sender: grSimSender = kwargs["sender"]
        movement = kwargs["movement"]

        rx, ry, rtheta = context.my_state.position
        ball_x, ball_y = context.ball.position
        ball_vx, ball_vy = context.ball.velocity

        opponents = list(context.opponents.values())

        best_target = None
        best_diff = float("inf")

        # Proactive: predict globally first
        for k in range(1, STEPS + 1):
            bx_f = ball_x + ball_vx * k * DT
            by_f = ball_y + ball_vy * k * DT

            # Check if this future point is safe from opponents
            opp_positions = [op.position for op in opponents]
            if not is_safe_from_robots(bx_f, by_f, opp_positions):
                continue

            dist_mm = np.hypot(bx_f - rx, by_f - ry)
            t_robot = time_to_reach(dist_mm)
            t_ball = k * DT

            diff = abs(t_robot - t_ball)
            if diff < best_diff:
                best_diff = diff
                best_target = (bx_f, by_f)

        if best_target is None:
            best_target = (ball_x, ball_y)

        tx, ty = best_target
        
        # Project to legal area after finding best global target to be proactive
        if self.role_area is not None:
            tx, ty = self.role_area.project_to_legal_area(context, tx, ty)
        
        obstacles = list(context.opponents.values()) + list(context.teammates.values())
        static_obs = [tuple(o.position[:2]) for o in obstacles]
        tx, ty = avoid_collision_robust(rx, ry, tx, ty, obstacles)

        vx, vy = movement.go_to(
            context.my_state.position,
            (tx, ty),
            max_speed=3000,
            static_obstacles=static_obs,
        )
        w = movement.face_towards(context.my_state.position, context.ball.position)

        sender.send_command(
            RobotCommand(context.robot_id, vx, vy, w, kick=0, dribble=1)
        )


       
class KickToGoalState(State[RobotContext]):
    """
    Striker-only shooting state.
    - Searches for open lane to goal (y: -400 to 400)
    - Clears lane
    - Aligns
    - Kicks
    """

    ALIGN_THRESHOLD_DEG = 2.5
    REPOSITION_SPEED = 1500

    def __init__(self, role_area: Optional[RoleArea] = None, is_goalie: bool = False, force_center: bool = False):
        self._enter_time = None
        self.role_area = role_area
        self.is_goalie = is_goalie
        self.force_center = force_center

    def handle_transition(self, context: RobotContext):
        # Role controls exit
        if not context.has_possession:
            self._enter_time = None
            return None
        
        if self._enter_time is None:
            self._enter_time = time.time()
            
        return None

    def execute(self, context: RobotContext, **kwargs):
        sender: grSimSender = kwargs["sender"]
        movement = kwargs["movement"]
        spawner: BallIdleSpawner = kwargs["spawner"]

        rx, ry, rtheta = context.my_state.position
        bx, by = context.ball.position
        
        # Check timeout - 5 seconds
        force_shoot = False
        if self._enter_time is not None and (time.time() - self._enter_time > 2.0):
            force_shoot = True

        # --------------------------------------------
        # 1. Pick open goal target
        # --------------------------------------------
        clear_target = pick_open_goal(context, spawner, force_center=self.force_center)
        
        # --------------------------------------------
        # 2. Align and Shoot
        # --------------------------------------------
        dx = clear_target[0] - bx
        dy = clear_target[1] - by
        goal_angle = math.atan2(dy, dx)

        angle_error = (goal_angle - rtheta + math.pi) % (2 * math.pi) - math.pi
        angle_error_deg = abs(math.degrees(angle_error))

        if angle_error_deg > self.ALIGN_THRESHOLD_DEG:
            w = movement.face_towards(
                robot_pos=context.my_state.position,
                target=clear_target,
                max_angular_speed=5.0,
                max_angular_acceleration=8.0,
            )
            
            # Reposition movement to stay in zone and align better
            target = self.role_area.project_to_legal_area(context, rx, ry, is_goalie=self.is_goalie) if self.role_area else np.array([rx, ry])
            
            # Always project to legal area to ensure boundary respect
            obstacles = list(context.opponents.values()) + list(context.teammates.values())
            static_obs = [tuple(o.position[:2]) for o in obstacles]
            vx, vy = movement.go_to(context.my_state.position, target, max_speed=self.REPOSITION_SPEED, static_obstacles=static_obs)
            
            # Keep dribbler on while aligning
            sender.send_command(
                RobotCommand(
                    context.robot_id,
                    vx=vx,
                    vy=vy,
                    w=w,
                    kick=0,
                    dribble=1,
                )
            )
            return

        # --------------------------------------------
        # 3. KICK
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
    def __init__(self, role_area: Optional[RoleArea] = None, is_goalie: bool = False, force_center: bool = False):
        super().__init__(initial_state=KickToGoalState(role_area=role_area, is_goalie=is_goalie, force_center=force_center))

class BlockingState(State[RobotContext]):
    """Stay between the opponent with the ball and our defense area."""
    def __init__(self, role_area: Optional[RoleArea] = None):
        self.role_area = role_area

    def handle_transition(self, context: RobotContext):
        return None

    def execute(self, context: RobotContext, **kwargs):
        sender: grSimSender = kwargs["sender"]
        movement = kwargs["movement"]
        spawner: BallIdleSpawner = kwargs["spawner"]

        # 1. Identify opponent with possession
        opponent_with_ball = None
        ball_pos = np.array(context.ball.position[:2])
        for op in context.opponents.values():
            if np.linalg.norm(op.position[:2] - ball_pos) < 150.0: # possession threshold
                opponent_with_ball = op
                break
        
        if opponent_with_ball is None:
            # Fallback to ball position
            target_source = ball_pos
        else:
            target_source = np.array(opponent_with_ball.position[:2])

        # 2. Define our goal center
        goal_sign = 1 if context.is_yellow else -1
        goal_x = goal_sign * spawner.goal_line_x
        goal_center = np.array([goal_x, 0.0])

        # 3. Position between opponent and goal
        # Stay at a distance that blocks the path to the defense area
        goal_to_opp = target_source - goal_center
        dist_to_opp = np.linalg.norm(goal_to_opp)
        
        if dist_to_opp > 500.0:
            # Position at 40% of the way from goal to opponent, but at least 1200mm from goal
            blocking_dist = max(1200.0, dist_to_opp * 0.4)
            # Ensure we don't crash into the opponent
            blocking_dist = min(blocking_dist, dist_to_opp - 400.0) 
            target = goal_center + (goal_to_opp / dist_to_opp) * blocking_dist
        else:
            target = goal_center + (goal_to_opp / (dist_to_opp + 1e-6)) * 500.0

        tx, ty = target[0], target[1]

        # 4. Stay in legal area and avoid collisions
        if self.role_area is not None:
            tx, ty = self.role_area.project_to_legal_area(context, tx, ty)

        obstacles = list(context.opponents.values()) + list(context.teammates.values())
        static_obs = [tuple(o.position[:2]) for o in obstacles]
        tx, ty = avoid_collision_robust(context.my_state.position[0], context.my_state.position[1], tx, ty, obstacles)

        vx, vy = movement.go_to(
            context.my_state.position,
            (tx, ty),
            max_speed=2500,
            static_obstacles=static_obs
        )
        w = movement.face_towards(context.my_state.position, target_source)

        sender.send_command(
            RobotCommand(context.robot_id, vx, vy, w, kick=0, dribble=0)
        )

class BlockingStateMachine(StateMachine[RobotContext]):
    def __init__(self, role_area: Optional[RoleArea] = None):
        super().__init__(initial_state=BlockingState(role_area))

class InterceptionStateMachine(StateMachine[RobotContext]):
    """Intercept fast balls or block passing lanes."""

    def __init__(self, role_area: RoleArea):
        intercept = InterceptFastBallState(role_area)
        block = BlockPassingLaneState(role_area)
        
        super().__init__(initial_state=block)
        self.register_state(intercept)


class HoldPositionFaceBallStateMachine(StateMachine):
    def __init__(self, home):
        super().__init__(HoldPositionFaceBallState(home))


class SupportBehindBallStateMachine(StateMachine):
    def __init__(self, yellow_goal):
        super().__init__(SupportBehindBallState(yellow_goal))   

class GoalieStateMachine(StateMachine):
    def __init__(self, role_area: Optional[RoleArea] = None):
        super().__init__(GoaliePositioningState(role_area))

class ShadowBallStateMachine(StateMachine):
    def __init__(self, role_area: RoleArea, margin: float = 300.0):
        super().__init__(ShadowBallState(role_area, margin=margin))