"""Shared ball handling state machines - reusable across roles."""

import time
from typing import Any, Optional

import numpy as np

from team_control.behavior.context import POSSESSION_THRESHOLD, EntityState, RobotContext
from team_control.motion.obstacles.helpers import avoid_collision_robust
from team_control.network.robot_command import RobotCommand
from team_control.network.ssl_sockets import grSimSender
from team_control.sim.ball_idle_spawner import BallIdleSpawner
from team_control.utils.state_machine import State, StateMachine

# Configuration Constants

# Distance thresholds (mm)
CHASE_TO_APPROACH_DISTANCE = 200  # Switch from fast chase to gentle approach
APPROACH_TO_CHASE_DISTANCE = 250  # Switch back to chase if ball moves away

# Speed and acceleration (mm/s and mm/s²)
CHASE_MAX_SPEED = 5000
CHASE_MAX_ACCELERATION = 1300
APPROACH_MAX_SPEED = 1500
APPROACH_MAX_ACCELERATION = 800
INTERCEPT_MAX_SPEED = 4000

# Passing alignment
ALIGNMENT_THRESHOLD = 0.05  # radians (~3 degrees)
KICK_POWER_MIN = 5  # grSim kick power units (for close range)
KICK_POWER_MAX = 9  # grSim kick power units (for long range)
KICK_DISTANCE_MIN = 2000  # mm - distance for minimum kick power
KICK_DISTANCE_MAX = 8000  # mm - distance for maximum kick power

# Receiving behavior
INTERCEPT_MIN_BALL_SPEED = 500  # mm/s - below this, don't intercept
INTERCEPT_THRESHOLD_SPEED = 800  # mm/s - speed at which interception becomes aggressive
INTERCEPT_DOT_THRESHOLD = 0.5  # Dot product threshold for "moving towards robot"

# Field margins
RECEIVER_FIELD_MARGIN = 200  # mm from field boundaries

# Angular acceleration
ALIGN_ANGULAR_ACCELERATION = 3  # rad/s²

# Repositioning constants
REPOSITION_STEP_SIZE = 800  # mm - how far to move when repositioning
REPOSITION_MAX_SPEED = 1500  # mm/s - speed during repositioning
SHADOW_WIDTH = 300  # mm - how wide the "blocked" zone is around the passing lane
CENTER_BIAS_STRENGTH = 0.3  # How strongly to pull toward center (0-1)
SEPARATION_BIAS_STRENGTH = 0.3  # How strongly to push away from teammate (0-1)
MIN_SEPARATION_DISTANCE = 2000  # mm - ideal minimum distance between robots

# Shooting at goal
SHOOTING_ALIGNMENT_THRESHOLD = 2  # radians (~2 degrees)
SHOOTING_KICK_POWER = 10  # Maximum power


# striker constraints
STRIKER_X_LIMIT = 1600.0
STRIKER_MARGIN = 120.0


def distance(pos0: np.ndarray, pos1: np.ndarray) -> float:
    """Calculate Euclidean distance between two 2D positions."""
    return float(np.linalg.norm(pos0[:2] - pos1[:2]))


# =============================================================================
# Chasing Ball States
# =============================================================================


class ChasingBallState(State[RobotContext]):
    """Fast approach to ball from distance."""

    def __init__(self, role_area: Optional[Any] = None, is_goalie: bool = False):
        self.role_area = role_area
        self.is_goalie = is_goalie

    def handle_transition(self, context: RobotContext) -> Optional[type[State[RobotContext]]]:
        ball_dist = distance(context.ball.position, context.my_state.position)
        if ball_dist < CHASE_TO_APPROACH_DISTANCE:
            return ApproachingBallState
        return None

    def execute(self, context: RobotContext, **kwargs: Any) -> None:
        sender: grSimSender = kwargs["sender"]
        movement = kwargs["movement"]

        target_x, target_y = context.ball.position[:2]

        # Avoid collision with other robots
        obstacles = list(context.opponents.values()) + list(context.teammates.values())
        target_x, target_y = avoid_collision_robust(
            context.my_state.position[0],
            context.my_state.position[1],
            target_x,
            target_y,
            obstacles,
        )

        if self.role_area is not None:
            target_x, target_y = self.role_area.project_to_legal_area(
                context, target_x, target_y, is_goalie=self.is_goalie
            )

        # Prepare obstacles for movement
        static_obs = [tuple(o.position[:2]) for o in context.opponents.values()] + [
            tuple(t.position[:2]) for t in context.teammates.values()
        ]

        vx, vy = movement.go_to(
            context.my_state.position,
            (target_x, target_y),
            kicker_to_target=True,
            max_acceleration=CHASE_MAX_ACCELERATION,
            max_speed=CHASE_MAX_SPEED,
            static_obstacles=static_obs,
        )
        w = movement.face_towards(robot_pos=context.my_state.position, target=context.ball.position)
        sender.send_command(
            RobotCommand(robot_id=context.robot_id, vx=vx, vy=vy, w=w, kick=0, dribble=1)
        )


class ApproachingBallState(State[RobotContext]):
    """Gentle approach when close to ball - prevents bumping."""

    def __init__(self, role_area: Optional[Any] = None, is_goalie: bool = False):
        self.role_area = role_area
        self.is_goalie = is_goalie

    def handle_transition(self, context: RobotContext) -> Optional[type[State[RobotContext]]]:
        ball_dist = distance(context.ball.position, context.my_state.position)

        if ball_dist > APPROACH_TO_CHASE_DISTANCE:
            return ChasingBallState
        # Role handles transition to PASSING/SHOOTING once possession is gained
        return None

    def execute(self, context: RobotContext, **kwargs: Any) -> None:
        sender: grSimSender = kwargs["sender"]
        movement = kwargs["movement"]

        target_x, target_y = context.ball.position[:2]

        # Avoid collision with other robots
        obstacles = list(context.opponents.values()) + list(context.teammates.values())
        target_x, target_y = avoid_collision_robust(
            context.my_state.position[0],
            context.my_state.position[1],
            target_x,
            target_y,
            obstacles,
        )

        if self.role_area is not None:
            target_x, target_y = self.role_area.project_to_legal_area(
                context, target_x, target_y, is_goalie=self.is_goalie
            )

        # Prepare obstacles for movement
        static_obs = [tuple(o.position[:2]) for o in context.opponents.values()] + [
            tuple(t.position[:2]) for t in context.teammates.values()
        ]

        vx, vy = movement.go_to(
            context.my_state.position,
            target=(target_x, target_y),
            kicker_to_target=True,
            max_acceleration=APPROACH_MAX_ACCELERATION,
            max_speed=APPROACH_MAX_SPEED,
            static_obstacles=static_obs,
        )

        w = movement.face_towards(robot_pos=context.my_state.position, target=context.ball.position)
        sender.send_command(
            RobotCommand(robot_id=context.robot_id, vx=vx, vy=vy, w=w, kick=0, dribble=1)
        )


class ChasingBallStateMachine(StateMachine[RobotContext]):
    """State machine for chasing and capturing the ball."""

    def __init__(self, role_area: Optional[Any] = None, is_goalie: bool = False):
        chasing = ChasingBallState(role_area=role_area, is_goalie=is_goalie)
        approaching = ApproachingBallState(role_area=role_area, is_goalie=is_goalie)

        super().__init__(initial_state=chasing)
        self.register_state(approaching)


# =============================================================================
# Passing States
# =============================================================================


def select_passing_target(context: RobotContext):
    """
    Select the best teammate to pass to.
    Respects preferred_teammate if set, otherwise picks most forward.
    """
    if not context.teammates:
        return None

    if context.preferred_teammate is not None and context.preferred_teammate in context.teammates:
        return context.teammates[context.preferred_teammate]

    # Most forward:
    # For Blue (playing right): max X
    # For Yellow (playing left): min X
    if context.is_yellow:
        return min(context.teammates.values(), key=lambda t: t.position[0])
    else:
        return max(context.teammates.values(), key=lambda t: t.position[0])


def find_teammate_with_ball(context: RobotContext) -> Optional[EntityState]:
    """Find the teammate who currently has possession of the ball."""
    for t_state in context.teammates.values():
        dist = np.linalg.norm(context.ball.position - t_state.position[:2])
        if dist < POSSESSION_THRESHOLD:
            return t_state
    return None


class AligningWithTargetState(State[RobotContext]):
    """Align with target to prepare for pass."""

    def __init__(self, role_area: Optional[Any] = None, is_goalie: bool = False):
        self._enter_time = None
        self.role_area = role_area
        self.is_goalie = is_goalie

    def handle_transition(self, context: RobotContext) -> Optional[type[State[RobotContext]]]:
        if not context.has_possession:
            self._enter_time = None
            return ApproachingBallState

        if self._enter_time is None:
            self._enter_time = time.time()

        # If stuck in passing/aligning for too long, switch to shooting
        if time.time() - self._enter_time > 2.0:
            return AligningWithGoalState

        # Get the intended target
        target = select_passing_target(context)
        if target is None:
            return None

        target_pos = target.position

        # Check if path is blocked by any opponent
        if not is_path_clear(context.my_state.position, target_pos, context.opponents.values()):
            # Don't transition to kicking if blocked
            return None

        # Calculate angle to target
        my_orientation = context.my_state.position[2]
        dx = target_pos[0] - context.my_state.position[0]
        dy = target_pos[1] - context.my_state.position[1]
        angle_to_target = np.arctan2(dy, dx)

        # Normalize angle difference to [-pi, pi]
        angle_diff = angle_to_target - my_orientation
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        # If aligned, transition to kicking
        if abs(angle_diff) < ALIGNMENT_THRESHOLD:
            return KickingState

        return None

    def execute(self, context: RobotContext, **kwargs: Any) -> None:
        sender: grSimSender = kwargs["sender"]
        movement = kwargs["movement"]
        spawner: BallIdleSpawner = kwargs["spawner"]

        target = select_passing_target(context)
        if target is None:
            return

        target_pos = target.position

        # Check if path is blocked - if so, move sideways to get clear
        vx, vy = 0, 0
        blocking_robot = None
        all_robots = list(context.opponents.values()) + list(context.teammates.values())
        for r in all_robots:
            if is_path_blocked(context.my_state.position, target_pos, r.position):
                blocking_robot = r
                break

        if blocking_robot:
            reposition_target = calculate_receiver_position(
                context.my_state.position,
                target_pos,
                blocking_robot,
                spawner.field_x_min,
                spawner.field_x_max,
                spawner.field_y_min,
                spawner.field_y_max,
            )
            if reposition_target is not None:
                tx, ty = reposition_target[0], reposition_target[1]
                if self.role_area is not None:
                    tx, ty = self.role_area.project_to_legal_area(
                        context, tx, ty, is_goalie=self.is_goalie
                    )

                obstacles = list(context.opponents.values()) + list(context.teammates.values())
                static_obs = [tuple(o.position[:2]) for o in obstacles]
                vx, vy = movement.go_to(
                    context.my_state.position,
                    (tx, ty),
                    max_speed=REPOSITION_MAX_SPEED,
                    static_obstacles=static_obs,
                )

        w = movement.face_towards(
            robot_pos=context.my_state.position,
            target=target_pos,
            max_angular_acceleration=ALIGN_ANGULAR_ACCELERATION,
        )

        sender.send_command(
            RobotCommand(robot_id=context.robot_id, vx=vx, vy=vy, w=w, kick=0, dribble=1)
        )


class KickingState(State[RobotContext]):
    """Execute the kick."""

    def handle_transition(self, context: RobotContext) -> Optional[type[State[RobotContext]]]:
        if not context.has_possession:
            return ChasingBallState
        return None

    def execute(self, context: RobotContext, **kwargs: Any) -> None:
        sender: grSimSender = kwargs["sender"]

        target = select_passing_target(context)
        if target is not None:
            target_pos = target.position
            dist = distance(context.my_state.position, target_pos)

            # Linear interpolation between min and max power
            t = (dist - KICK_DISTANCE_MIN) / (KICK_DISTANCE_MAX - KICK_DISTANCE_MIN)
            t = max(0.0, min(1.0, t))  # Clamp to [0, 1]
            kick_power = KICK_POWER_MIN + t * (KICK_POWER_MAX - KICK_POWER_MIN)
        else:
            kick_power = KICK_POWER_MIN

        sender.send_command(
            RobotCommand(robot_id=context.robot_id, vx=0, vy=0, w=0, kick=kick_power, dribble=0)
        )


class PassingStateMachine(StateMachine[RobotContext]):
    """State machine for passing the ball to a teammate."""

    def __init__(self, role_area: Optional[Any] = None, is_goalie: bool = False):
        aligning = AligningWithTargetState(role_area=role_area, is_goalie=is_goalie)
        kicking = KickingState()
        approaching = ApproachingBallState(
            role_area=role_area, is_goalie=is_goalie
        )  # Fallback if possession lost
        chasing = ChasingBallState(role_area=role_area, is_goalie=is_goalie)  # Fallback if ball far
        align_goal = AligningWithGoalState()
        shooting = ShootingState()

        super().__init__(initial_state=aligning)
        self.register_state(kicking)
        self.register_state(approaching)
        self.register_state(chasing)
        self.register_state(align_goal)
        self.register_state(shooting)


# =============================================================================
# Receiving States
# =============================================================================


def should_intercept(
    ball_pos: np.ndarray, ball_velocity: np.ndarray, robot_pos: np.ndarray
) -> bool:
    """Check if ball is moving toward robot and should be intercepted."""
    ball_speed = np.linalg.norm(ball_velocity)

    if ball_speed < INTERCEPT_MIN_BALL_SPEED:
        return False

    if ball_speed < INTERCEPT_THRESHOLD_SPEED:
        return False

    # Check if ball moving toward robot
    ball_direction = ball_velocity / ball_speed
    to_robot = robot_pos[:2] - ball_pos
    to_robot_dist = np.linalg.norm(to_robot)

    if to_robot_dist == 0:
        return False

    to_robot_normalized = to_robot / to_robot_dist
    dot = np.dot(ball_direction, to_robot_normalized)

    return dot > INTERCEPT_DOT_THRESHOLD


def calculate_interception_point(
    ball_pos: np.ndarray, ball_velocity: np.ndarray, robot_pos: np.ndarray
) -> np.ndarray:
    """Calculate closest point on ball trajectory for interception.

    Projects the robot's position onto the ball's trajectory line to find
    the optimal interception point (perpendicular to the trajectory).
    """
    ball_speed = np.linalg.norm(ball_velocity)
    if ball_speed == 0:
        return ball_pos

    # Direction of ball movement (normalized)
    ball_direction = ball_velocity / ball_speed

    # Vector from ball to robot
    ball_to_robot = robot_pos[:2] - ball_pos

    # Project robot position onto the ball's trajectory line
    # This gives the distance along the trajectory to the closest point
    projection_distance = np.dot(ball_to_robot, ball_direction)

    # Only intercept points ahead of the ball (positive projection)
    if projection_distance < 0:
        # Ball is moving away from robot, just go to current ball position
        return ball_pos

    # Calculate the closest point on the trajectory
    interception_point = ball_pos + ball_direction * projection_distance

    return interception_point


def is_path_blocked(
    passer_pos: np.ndarray,
    receiver_pos: np.ndarray,
    opponent_pos: np.ndarray,
) -> bool:
    """Check if opponent is blocking the passing lane between passer and receiver."""
    # Vector from passer to receiver
    passer_to_receiver = receiver_pos[:2] - passer_pos[:2]
    lane_length = np.linalg.norm(passer_to_receiver)

    if lane_length < 1:
        return False

    lane_dir = passer_to_receiver / lane_length

    # Vector from passer to opponent
    passer_to_opponent = opponent_pos[:2] - passer_pos[:2]

    # Project opponent onto the passing lane
    projection = np.dot(passer_to_opponent, lane_dir)

    # Opponent must be between passer and receiver (with some margin)
    if projection < 200 or projection > lane_length - 200:
        return False

    # Calculate perpendicular distance from opponent to the lane
    closest_point_on_lane = passer_pos[:2] + lane_dir * projection
    perp_distance = np.linalg.norm(opponent_pos[:2] - closest_point_on_lane)

    return perp_distance < SHADOW_WIDTH


def is_path_clear(
    passer_pos: np.ndarray,
    receiver_pos: np.ndarray,
    opponents,
) -> bool:
    """
    True if no opponent blocks the passing lane.
    """
    for opp in opponents:
        if is_path_blocked(passer_pos, receiver_pos, opp.position):
            return False
    return True


def calculate_receiver_position(
    receiver_pos: np.ndarray,
    passer_pos: np.ndarray,
    opponent_state: Optional["EntityState"],
    field_x_min: float,
    field_x_max: float,
    field_y_min: float,
    field_y_max: float,
    margin: float = RECEIVER_FIELD_MARGIN,
) -> Optional[np.ndarray]:
    """Calculate a new position to get clear of opponent's shadow.

    Only moves when path is actually blocked. Moves toward the side with
    more open field space.
    Returns None if already in a good position (no movement needed).
    """
    # If no opponent, stay put
    if opponent_state is None:
        return None

    opponent_pos = opponent_state.position

    # Only move if path is actually blocked - otherwise stay completely still
    if not is_path_blocked(passer_pos, receiver_pos, opponent_pos):
        return None

    # Path is blocked - need to move perpendicular to clear the lane
    passer_to_receiver = receiver_pos[:2] - passer_pos[:2]
    lane_length = np.linalg.norm(passer_to_receiver)

    if lane_length < 1:
        return None

    lane_dir = passer_to_receiver / lane_length

    # Perpendicular directions (left and right of the lane)
    perp_left = np.array([-lane_dir[1], lane_dir[0]])
    perp_right = np.array([lane_dir[1], -lane_dir[0]])

    # Calculate how much space is available in each direction
    left_candidate = receiver_pos[:2] + perp_left * REPOSITION_STEP_SIZE
    right_candidate = receiver_pos[:2] + perp_right * REPOSITION_STEP_SIZE

    # Choose direction based purely on available space
    def space_available(pos: np.ndarray) -> float:
        return min(
            pos[0] - field_x_min,
            field_x_max - pos[0],
            pos[1] - field_y_min,
            field_y_max - pos[1],
        )

    left_space = space_available(left_candidate)
    right_space = space_available(right_candidate)

    if left_space > right_space:
        candidate = left_candidate
    else:
        candidate = right_candidate

    # Add separate center bias: pull toward center proportional to distance
    field_center = np.array([(field_x_min + field_x_max) / 2, (field_y_min + field_y_max) / 2])
    to_center = field_center - receiver_pos[:2]
    dist_to_center = np.linalg.norm(to_center)

    if dist_to_center > 100:  # Only apply if not already at center
        # Normalize and scale by distance (further = stronger pull)
        center_dir = to_center / dist_to_center
        # Scale factor: at 3000mm from center, add full CENTER_BIAS_STRENGTH
        center_pull = (
            center_dir * (dist_to_center / 3000) * CENTER_BIAS_STRENGTH * REPOSITION_STEP_SIZE
        )
        candidate = candidate + center_pull

    # Add separation bias: push away from passer if too close
    to_passer = passer_pos[:2] - receiver_pos[:2]
    dist_to_passer = np.linalg.norm(to_passer)

    if dist_to_passer < MIN_SEPARATION_DISTANCE and dist_to_passer > 100:
        # Push away from passer, stronger when closer
        away_from_passer = -to_passer / dist_to_passer
        # Scale: at half the min distance, apply full strength
        closeness_factor = (MIN_SEPARATION_DISTANCE - dist_to_passer) / MIN_SEPARATION_DISTANCE
        separation_push = (
            away_from_passer * closeness_factor * SEPARATION_BIAS_STRENGTH * REPOSITION_STEP_SIZE
        )
        candidate = candidate + separation_push

    # Clamp to field boundaries
    candidate = np.array(
        [
            max(field_x_min + margin, min(field_x_max - margin, candidate[0])),
            max(field_y_min + margin, min(field_y_max - margin, candidate[1])),
        ]
    )

    return candidate


class PositioningState(State[RobotContext]):
    """Position to receive pass."""

    def __init__(self, role_area: Optional[Any] = None, is_goalie: bool = False):
        self.role_area = role_area
        self.is_goalie = is_goalie

    def handle_transition(self, context: RobotContext) -> Optional[type[State[RobotContext]]]:
        if should_intercept(
            context.ball.position, context.ball.velocity, context.my_state.position
        ):
            return InterceptingState
        return None

    def execute(self, context: RobotContext, **kwargs: Any) -> None:
        sender: grSimSender = kwargs["sender"]
        movement = kwargs["movement"]
        spawner: BallIdleSpawner = kwargs["spawner"]

        # Identify the likely passer (teammate with ball or closest to ball)
        passer_state = find_teammate_with_ball(context)
        if passer_state is None and context.teammates:
            passer_state = min(
                context.teammates.values(),
                key=lambda t: np.linalg.norm(context.ball.position - t.position[:2]),
            )

        if passer_state is None:
            return

        passer_pos = passer_state.position

        ball_speed = np.linalg.norm(context.ball.velocity)

        # If ball is moving (even slowly), move to intercept its trajectory
        if ball_speed > INTERCEPT_MIN_BALL_SPEED:
            # Check if ball is moving somewhat towards us
            ball_direction = context.ball.velocity / ball_speed
            to_robot = context.my_state.position[:2] - context.ball.position
            to_robot_dist = np.linalg.norm(to_robot)

            if to_robot_dist > 0:
                to_robot_normalized = to_robot / to_robot_dist
                dot = np.dot(ball_direction, to_robot_normalized)

                # If ball is moving towards robot (even loosely), intercept trajectory
                if dot > 0.1:  # Lower threshold than should_intercept
                    target_pos = calculate_interception_point(
                        context.ball.position, context.ball.velocity, context.my_state.position
                    )
                    tx, ty = target_pos[0], target_pos[1]

                    # Avoid collision with other robots
                    obstacles = list(context.opponents.values()) + list(context.teammates.values())
                    tx, ty = avoid_collision_robust(
                        context.my_state.position[0],
                        context.my_state.position[1],
                        tx,
                        ty,
                        obstacles,
                    )

                    if self.role_area is not None:
                        tx, ty = self.role_area.project_to_legal_area(
                            context, tx, ty, is_goalie=self.is_goalie
                        )

                    # Prepare obstacles for movement
                    static_obs = [tuple(o.position[:2]) for o in context.opponents.values()] + [
                        tuple(t.position[:2]) for t in context.teammates.values()
                    ]

                    vx, vy = movement.go_to(
                        context.my_state.position,
                        (tx, ty),
                        kicker_to_target=True,
                        static_obstacles=static_obs,
                    )
                    w = movement.face_towards(context.my_state.position, context.ball.position)
                    sender.send_command(
                        RobotCommand(
                            robot_id=context.robot_id, vx=vx, vy=vy, w=w, kick=0, dribble=1
                        )
                    )
                    return

        # Check for other robots blocking the passing lane
        blocking_robot = None
        all_robots = list(context.opponents.values()) + list(context.teammates.values())
        for r in all_robots:
            if is_path_blocked(passer_pos, context.my_state.position, r.position):
                blocking_robot = r
                break

        # Check if we need to reposition to avoid other robot's shadow
        target_pos = calculate_receiver_position(
            context.my_state.position,
            passer_pos,
            blocking_robot,
            spawner.field_x_min,
            spawner.field_x_max,
            spawner.field_y_min,
            spawner.field_y_max,
        )

        if target_pos is not None:
            tx, ty = target_pos[0], target_pos[1]
            if self.role_area is not None:
                tx, ty = self.role_area.project_to_legal_area(
                    context, tx, ty, is_goalie=self.is_goalie
                )
            # Prepare obstacles for movement
            static_obs = [tuple(o.position[:2]) for o in context.opponents.values()] + [
                tuple(t.position[:2]) for t in context.teammates.values()
            ]

            # Need to move to get clear
            vx, vy = movement.go_to(
                context.my_state.position,
                (tx, ty),
                max_speed=REPOSITION_MAX_SPEED,
                static_obstacles=static_obs,
            )
        else:
            # Already in good position, stay still
            vx, vy = 0, 0

        w = movement.face_towards(context.my_state.position, passer_pos)

        sender.send_command(
            RobotCommand(robot_id=context.robot_id, vx=vx, vy=vy, w=w, kick=0, dribble=0)
        )


class InterceptingState(State[RobotContext]):
    """Intercept incoming ball."""

    def __init__(self, role_area: Optional[Any] = None, is_goalie: bool = False):
        self.role_area = role_area
        self.is_goalie = is_goalie

    def handle_transition(self, context: RobotContext) -> Optional[type[State[RobotContext]]]:
        # Calculate distance to the interception point on the trajectory
        interception_point = calculate_interception_point(
            context.ball.position, context.ball.velocity, context.my_state.position
        )
        dist_to_interception = np.linalg.norm(interception_point - context.my_state.position[:2])

        # Also check distance to ball itself
        dist_to_ball = np.linalg.norm(context.ball.position - context.my_state.position[:2])

        # Keep intercepting if:
        # 1. Close to interception point, OR
        # 2. Close to the ball itself
        # Only exit if both are far away and ball is not coming toward us
        if dist_to_interception < 500 or dist_to_ball < 500:
            return None  # Stay in intercepting

        if not should_intercept(
            context.ball.position, context.ball.velocity, context.my_state.position
        ):
            return PositioningState
        return None

    def execute(self, context: RobotContext, **kwargs: Any) -> None:
        sender: grSimSender = kwargs["sender"]
        movement = kwargs["movement"]

        interception_point = calculate_interception_point(
            context.ball.position, context.ball.velocity, context.my_state.position
        )

        tx, ty = interception_point[0], interception_point[1]

        # Avoid collision with other robots
        obstacles = list(context.opponents.values()) + list(context.teammates.values())
        tx, ty = avoid_collision_robust(
            context.my_state.position[0], context.my_state.position[1], tx, ty, obstacles
        )

        if self.role_area is not None:
            tx, ty = self.role_area.project_to_legal_area(context, tx, ty, is_goalie=self.is_goalie)

        # Prepare obstacles for movement
        static_obs = [tuple(o.position[:2]) for o in context.opponents.values()] + [
            tuple(t.position[:2]) for t in context.teammates.values()
        ]

        vx, vy = movement.go_to(
            context.my_state.position,
            (tx, ty),
            kicker_to_target=True,
            max_speed=INTERCEPT_MAX_SPEED,
            static_obstacles=static_obs,
        )
        w = movement.face_towards(context.my_state.position, context.ball.position)

        sender.send_command(
            RobotCommand(robot_id=context.robot_id, vx=vx, vy=vy, w=w, kick=0, dribble=1)
        )


def pick_open_goal(
    context: RobotContext,
    spawner,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Returns:
      (goal_target, reposition_target)
    - goal_target: midpoint of goal
    - reposition_target: where to move if blocked
    """

    ball_xy = context.ball.position[:2]
    gx = -4500.0 if context.is_yellow else 4500.0
    goal_target = np.array([gx, 0.0], dtype=float)

    opponent = list(context.opponents.values())[0] if context.opponents else None

    reposition = calculate_receiver_position(
        receiver_pos=ball_xy,
        passer_pos=goal_target,  # ← goal acts like "passer"
        opponent_state=opponent,
        field_x_min=spawner.field_x_min,
        field_x_max=spawner.field_x_max,
        field_y_min=spawner.field_y_min,
        field_y_max=spawner.field_y_max,
    )

    # None means NOT BLOCKED → shoot
    if reposition is None:
        return goal_target, None

    return None, reposition


class StrikerPositioningState(PositioningState):
    def execute(self, context: RobotContext, **kwargs):
        sender = kwargs["sender"]
        movement = kwargs["movement"]
        spawner = kwargs["spawner"]

        if not context.has_possession:
            return  # super().execute(context, **kwargs)

        # -------- STRIKER SHOOT DECISION --------
        goal_target, reposition = pick_open_goal(context, spawner)

        # BLOCKED → reposition laterally
        if goal_target is None:
            # Prepare obstacles for movement
            static_obs = [tuple(o.position[:2]) for o in context.opponents.values()] + [
                tuple(t.position[:2]) for t in context.teammates.values()
            ]

            vx, vy = movement.go_to(
                context.my_state.position,
                reposition,
                max_speed=1500,
                static_obstacles=static_obs,
            )

            w = movement.face_towards(context.my_state.position, reposition)
            sender.send_command(RobotCommand(context.robot_id, vx, vy, w, kick=0, dribble=1))
            return

        # CLEAR → let Aligning/Kicking handle the shot
        return super().execute(context, **kwargs)


class ReceivingStateMachine(StateMachine[RobotContext]):
    """State machine for receiving passes."""

    def __init__(self, role_area: Optional[Any] = None, is_goalie: bool = False):
        positioning = PositioningState(role_area=role_area, is_goalie=is_goalie)
        intercepting = InterceptingState(role_area=role_area, is_goalie=is_goalie)

        super().__init__(initial_state=positioning)
        self.register_state(intercepting)


class StrikerStateMachine(StateMachine):
    def __init__(self, role_area: Optional[Any] = None):
        positioning = StrikerPositioningState(role_area=role_area)
        aligning = AligningWithTargetState(role_area=role_area)
        kicking = KickingState()

        super().__init__(initial_state=positioning)
        self.register_state(aligning)
        self.register_state(kicking)


# =============================================================================
# Shooting at Goal States
# =============================================================================


def find_goal_target(
    context: RobotContext, spawner: BallIdleSpawner, force_center: bool = False
) -> np.ndarray:
    """Find best target point in opponent's goal (searching for open lanes)."""
    goal_x = -spawner.goal_line_x if context.is_yellow else spawner.goal_line_x

    if force_center:
        return np.array([goal_x, 0.0])

    # Range of y in goal area: -400 to 400
    y_points = np.linspace(-400, 400, 59)

    ball_xy = context.ball.position[:2]
    opponents = list(context.opponents.values())

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

    return np.array([goal_x, 0.0])


class AligningWithGoalState(State[RobotContext]):
    """Align with goal to prepare for shot."""

    def __init__(
        self, role_area: Optional[Any] = None, is_goalie: bool = False, force_center: bool = False
    ):
        self.role_area = role_area
        self.is_goalie = is_goalie
        self.force_center = force_center

    def handle_transition(self, context: RobotContext) -> Optional[type[State[RobotContext]]]:
        if not context.has_possession:
            return ApproachingBallForShotState

        # Get goal target
        # Note: spawner is not available in handle_transition, so we use a simpler check
        goal_x = -4500 if context.is_yellow else 4500  # Approximate goal line

        my_orientation = context.my_state.position[2]
        dx = goal_x - context.my_state.position[0]
        dy = 0 - context.my_state.position[1]  # Aim at center for transition check
        angle_to_goal = np.arctan2(dy, dx)

        angle_diff = angle_to_goal - my_orientation
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        if abs(angle_diff) < SHOOTING_ALIGNMENT_THRESHOLD:
            return ShootingState

        return None

    def execute(self, context: RobotContext, **kwargs: Any) -> None:
        sender: grSimSender = kwargs["sender"]
        movement = kwargs["movement"]
        spawner: BallIdleSpawner = kwargs["spawner"]

        target_pos = find_goal_target(context, spawner, force_center=self.force_center)

        w = movement.face_towards(
            robot_pos=context.my_state.position,
            target=target_pos,
            max_angular_acceleration=ALIGN_ANGULAR_ACCELERATION,
        )

        # Ensure we stay in legal area while aligning
        rx, ry = context.my_state.position[:2]
        vx, vy = 0.0, 0.0
        if self.role_area is not None:
            target = self.role_area.project_to_legal_area(context, rx, ry, is_goalie=self.is_goalie)

            # Prepare obstacles for movement
            static_obs = [tuple(o.position[:2]) for o in context.opponents.values()] + [
                tuple(t.position[:2]) for t in context.teammates.values()
            ]

            vx, vy = movement.go_to(
                context.my_state.position, target, max_speed=500, static_obstacles=static_obs
            )

        sender.send_command(
            RobotCommand(robot_id=context.robot_id, vx=vx, vy=vy, w=w, kick=0, dribble=1)
        )


class ShootingState(State[RobotContext]):
    """Execute shot at goal with maximum power."""

    def handle_transition(self, context: RobotContext) -> Optional[type[State[RobotContext]]]:
        if not context.has_possession:
            return ChasingBallForShotState
        return None

    def execute(self, context: RobotContext, **kwargs: Any) -> None:
        sender: grSimSender = kwargs["sender"]

        sender.send_command(
            RobotCommand(
                robot_id=context.robot_id, vx=0, vy=0, w=0, kick=SHOOTING_KICK_POWER, dribble=0
            )
        )


class ChasingBallForShotState(State[RobotContext]):
    """Fast approach to ball for shooting."""

    def __init__(self, role_area: Optional[Any] = None, is_goalie: bool = False):
        self.role_area = role_area
        self.is_goalie = is_goalie

    def handle_transition(self, context: RobotContext) -> Optional[type[State[RobotContext]]]:
        ball_dist = distance(context.ball.position, context.my_state.position)
        if ball_dist < CHASE_TO_APPROACH_DISTANCE:
            return ApproachingBallForShotState
        if context.has_possession:
            return AligningWithGoalState
        return None

    def execute(self, context: RobotContext, **kwargs: Any) -> None:
        sender: grSimSender = kwargs["sender"]
        movement = kwargs["movement"]

        target_x, target_y = context.ball.position[:2]

        # Avoid collision with other robots
        obstacles = list(context.opponents.values()) + list(context.teammates.values())
        target_x, target_y = avoid_collision_robust(
            context.my_state.position[0],
            context.my_state.position[1],
            target_x,
            target_y,
            obstacles,
        )

        if self.role_area is not None:
            target_x, target_y = self.role_area.project_to_legal_area(
                context, target_x, target_y, is_goalie=self.is_goalie
            )

        # Prepare obstacles for movement
        static_obs = [tuple(o.position[:2]) for o in context.opponents.values()] + [
            tuple(t.position[:2]) for t in context.teammates.values()
        ]

        vx, vy = movement.go_to(
            context.my_state.position,
            (target_x, target_y),
            kicker_to_target=True,
            max_acceleration=CHASE_MAX_ACCELERATION,
            max_speed=CHASE_MAX_SPEED,
            static_obstacles=static_obs,
        )
        w = movement.face_towards(robot_pos=context.my_state.position, target=context.ball.position)
        sender.send_command(
            RobotCommand(robot_id=context.robot_id, vx=vx, vy=vy, w=w, kick=0, dribble=1)
        )


class ApproachingBallForShotState(State[RobotContext]):
    """Gentle approach when close to ball for shooting."""

    def __init__(self, role_area: Optional[Any] = None, is_goalie: bool = False):
        self.role_area = role_area
        self.is_goalie = is_goalie

    def handle_transition(self, context: RobotContext) -> Optional[type[State[RobotContext]]]:
        ball_dist = distance(context.ball.position, context.my_state.position)

        if ball_dist > APPROACH_TO_CHASE_DISTANCE:
            return ChasingBallForShotState
        if context.has_possession:
            return AligningWithGoalState
        return None

    def execute(self, context: RobotContext, **kwargs: Any) -> None:
        sender: grSimSender = kwargs["sender"]
        movement = kwargs["movement"]

        target_x, target_y = context.ball.position[:2]

        # Avoid collision with other robots
        obstacles = list(context.opponents.values()) + list(context.teammates.values())
        target_x, target_y = avoid_collision_robust(
            context.my_state.position[0],
            context.my_state.position[1],
            target_x,
            target_y,
            obstacles,
        )

        if self.role_area is not None:
            target_x, target_y = self.role_area.project_to_legal_area(
                context, target_x, target_y, is_goalie=self.is_goalie
            )

        # Prepare obstacles for movement
        static_obs = [tuple(o.position[:2]) for o in context.opponents.values()] + [
            tuple(t.position[:2]) for t in context.teammates.values()
        ]

        vx, vy = movement.go_to(
            context.my_state.position,
            (target_x, target_y),
            kicker_to_target=True,
            max_acceleration=APPROACH_MAX_ACCELERATION,
            max_speed=APPROACH_MAX_SPEED,
            static_obstacles=static_obs,
        )
        w = movement.face_towards(robot_pos=context.my_state.position, target=context.ball.position)
        sender.send_command(
            RobotCommand(robot_id=context.robot_id, vx=vx, vy=vy, w=w, kick=0, dribble=1)
        )


class ShootingStateMachine(StateMachine[RobotContext]):
    """State machine for shooting at goal."""

    def __init__(
        self, role_area: Optional[Any] = None, is_goalie: bool = False, force_center: bool = False
    ):
        chasing = ChasingBallForShotState(role_area=role_area, is_goalie=is_goalie)
        approaching = ApproachingBallForShotState(role_area=role_area, is_goalie=is_goalie)
        aligning = AligningWithGoalState(
            role_area=role_area, is_goalie=is_goalie, force_center=force_center
        )
        shooting = ShootingState()

        super().__init__(initial_state=aligning)
        self.register_state(chasing)
        self.register_state(approaching)
        self.register_state(shooting)
