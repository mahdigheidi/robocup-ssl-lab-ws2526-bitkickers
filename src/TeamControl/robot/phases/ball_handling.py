"""Shared ball handling state machines - reusable across roles."""

import numpy as np
from typing import Optional, Type, Any
from TeamControl.utils.state_machine import State, StateMachine
from TeamControl.network.ssl_sockets import grSimSender
from TeamControl.network.robot_command import RobotCommand
from TeamControl.robot.context import RobotContext
from TeamControl.ball.ball_idle_spawner import BallIdleSpawner


# =============================================================================
# Configuration Constants
# =============================================================================

# Distance thresholds (mm)
CHASE_TO_APPROACH_DISTANCE = 200  # Switch from fast chase to gentle approach
APPROACH_TO_CHASE_DISTANCE = 250  # Switch back to chase if ball moves away

# Speed and acceleration (mm/s and mm/s²)
CHASE_MAX_SPEED = 5000
CHASE_MAX_ACCELERATION = 1300
APPROACH_MAX_SPEED = 1500
APPROACH_MAX_ACCELERATION = 800
INTERCEPT_MAX_SPEED = 3000

# Passing alignment
ALIGNMENT_THRESHOLD = 0.1  # radians (~5.7 degrees)
KICK_POWER = 5  # grSim kick power units

# Receiving behavior
INTERCEPT_MIN_BALL_SPEED = 500  # mm/s - below this, don't intercept
INTERCEPT_THRESHOLD_SPEED = 800  # mm/s - speed at which interception becomes aggressive
INTERCEPT_DOT_THRESHOLD = 0.5  # Dot product threshold for "moving towards robot"
INTERCEPT_SHIFT_FACTOR = 0.8  # How far back along trajectory to position

# Field margins
RECEIVER_FIELD_MARGIN = 200  # mm from field boundaries

# Angular acceleration
ALIGN_ANGULAR_ACCELERATION = 3  # rad/s²


def distance(pos0: np.ndarray, pos1: np.ndarray) -> float:
    """Calculate Euclidean distance between two 2D positions."""
    return float(np.linalg.norm(pos0[:2] - pos1[:2]))


# =============================================================================
# Chasing Ball States
# =============================================================================

class ChasingBallState(State[RobotContext]):
    """Fast approach to ball from distance."""

    def handle_transition(self, context: RobotContext) -> Optional[Type[State[RobotContext]]]:
        ball_dist = distance(context.ball.position, context.my_state.position)
        if ball_dist < CHASE_TO_APPROACH_DISTANCE:
            return ApproachingBallState
        return None

    def execute(self, context: RobotContext, **kwargs: Any) -> None:
        sender: grSimSender = kwargs["sender"]
        movement = kwargs["movement"]
        
        vx, vy = movement.go_to(
            context.my_state.position,
            context.ball.position,
            kicker_to_target=True,
            max_acceleration=CHASE_MAX_ACCELERATION,
            max_speed=CHASE_MAX_SPEED,
        )
        w = movement.face_towards(
            robot_pos=context.my_state.position, target=context.ball.position
        )
        sender.send_command(
            RobotCommand(robot_id=context.robot_id, vx=vx, vy=vy, w=w, kick=0, dribble=1)
        )


class ApproachingBallState(State[RobotContext]):
    """Gentle approach when close to ball - prevents bumping."""

    def handle_transition(self, context: RobotContext) -> Optional[Type[State[RobotContext]]]:
        ball_dist = distance(context.ball.position, context.my_state.position)
        
        if ball_dist > APPROACH_TO_CHASE_DISTANCE:
            return ChasingBallState
        if context.has_possession:
            return AligningWithTargetState
        return None

    def execute(self, context: RobotContext, **kwargs: Any) -> None:
        sender: grSimSender = kwargs["sender"]
        movement = kwargs["movement"]
        
        vx, vy = movement.go_to(
            context.my_state.position,
            context.ball.position,
            kicker_to_target=True,
            max_acceleration=APPROACH_MAX_ACCELERATION,
            max_speed=APPROACH_MAX_SPEED,
        )
        w = movement.face_towards(
            robot_pos=context.my_state.position, target=context.ball.position
        )
        sender.send_command(
            RobotCommand(robot_id=context.robot_id, vx=vx, vy=vy, w=w, kick=0, dribble=1)
        )


class ChasingBallStateMachine(StateMachine[RobotContext]):
    """State machine for chasing and capturing the ball."""
    
    def __init__(self):
        chasing = ChasingBallState()
        approaching = ApproachingBallState()
        aligning = AligningWithTargetState()
        
        super().__init__(initial_state=chasing)
        self.register_state(approaching)
        self.register_state(aligning)


# =============================================================================
# Passing States
# =============================================================================

class AligningWithTargetState(State[RobotContext]):
    """Align with target to prepare for pass."""

    def handle_transition(self, context: RobotContext) -> Optional[Type[State[RobotContext]]]:
        if not context.has_possession:
            return ApproachingBallState
        
        # Check if aligned with target directly
        if not context.teammates:
            return None
        
        target_pos = list(context.teammates.values())[0].position
        
        # Calculate angle to target
        my_orientation = context.my_state.position[2]  # Current orientation
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
        
        # Get first teammate as passing target
        if not context.teammates:
            return
        
        target_pos = list(context.teammates.values())[0].position
        
        vx, vy = 0, 0
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

    def handle_transition(self, context: RobotContext) -> Optional[Type[State[RobotContext]]]:
        if not context.has_possession:
            return ChasingBallState
        return None

    def execute(self, context: RobotContext, **kwargs: Any) -> None:
        sender: grSimSender = kwargs["sender"]
        
        sender.send_command(
            RobotCommand(robot_id=context.robot_id, vx=0, vy=0, w=0, kick=KICK_POWER, dribble=0)
        )


class PassingStateMachine(StateMachine[RobotContext]):
    """State machine for passing the ball to a teammate."""
    
    def __init__(self):
        aligning = AligningWithTargetState()
        kicking = KickingState()
        approaching = ApproachingBallState()  # Fallback if possession lost
        chasing = ChasingBallState()  # Fallback if ball far
        
        super().__init__(initial_state=aligning)
        self.register_state(kicking)
        self.register_state(approaching)
        self.register_state(chasing)


# =============================================================================
# Receiving States
# =============================================================================

def should_intercept(ball_pos: np.ndarray, ball_velocity: np.ndarray, robot_pos: np.ndarray) -> bool:
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


def calculate_interception_point(ball_pos: np.ndarray, ball_velocity: np.ndarray) -> np.ndarray:
    """Calculate where to position to intercept ball."""
    ball_speed = np.linalg.norm(ball_velocity)
    if ball_speed == 0:
        return ball_pos
    
    # Project backward along trajectory
    ball_direction = ball_velocity / ball_speed
    shift_distance = INTERCEPT_SHIFT_FACTOR * ball_speed
    interception_point = ball_pos - ball_direction * shift_distance
    
    return interception_point


def calculate_receiver_position(
    passer_pos: np.ndarray,
    receiver_pos: np.ndarray,
    min_distance: float,
    field_x_min: float,
    field_x_max: float,
    field_y_min: float,
    field_y_max: float,
    margin: float = RECEIVER_FIELD_MARGIN,
) -> np.ndarray:
    """Calculate position for receiver at minimum distance from passer."""
    delta = receiver_pos[:2] - passer_pos[:2]
    dist = np.linalg.norm(delta)

    if dist == 0 or dist >= min_distance:
        return receiver_pos[:2]

    # Calculate target position at min_distance
    target = passer_pos[:2] + (delta / dist) * min_distance

    # Clamp to field boundaries
    clamped = np.array(
        [
            max(field_x_min + margin, min(field_x_max - margin, target[0])),
            max(field_y_min + margin, min(field_y_max - margin, target[1])),
        ]
    )

    # If clamping brought us closer than min_distance, project back
    clamped_delta = clamped - passer_pos[:2]
    clamped_dist = np.linalg.norm(clamped_delta)

    if clamped_dist < min_distance:
        if clamped_dist > 0:
            target = passer_pos[:2] + (clamped_delta / clamped_dist) * min_distance
        else:
            target = clamped
    else:
        target = clamped

    return target


class PositioningState(State[RobotContext]):
    """Position to receive pass."""

    def handle_transition(self, context: RobotContext) -> Optional[Type[State[RobotContext]]]:
        if should_intercept(context.ball.position, context.ball.velocity, context.my_state.position):
            return InterceptingState
        return None

    def execute(self, context: RobotContext, **kwargs: Any) -> None:
        sender: grSimSender = kwargs["sender"]
        movement = kwargs["movement"]
        spawner: BallIdleSpawner = kwargs["spawner"]
        passing_min_distance: float = kwargs.get("passing_min_distance", 3000)
        
        # Get first teammate position (for two-robot passing)
        if not context.teammates:
            return
        
        passer_pos = list(context.teammates.values())[0].position
        
        target_pos = calculate_receiver_position(
            passer_pos,
            context.my_state.position,
            passing_min_distance,
            spawner.field_x_min,
            spawner.field_x_max,
            spawner.field_y_min,
            spawner.field_y_max,
        )
        
        vx, vy = movement.go_to(context.my_state.position, target_pos)
        w = movement.face_towards(context.my_state.position, passer_pos)
        
        sender.send_command(
            RobotCommand(robot_id=context.robot_id, vx=vx, vy=vy, w=w, kick=0, dribble=0)
        )


class InterceptingState(State[RobotContext]):
    """Intercept incoming ball."""

    def handle_transition(self, context: RobotContext) -> Optional[Type[State[RobotContext]]]:
        if not should_intercept(context.ball.position, context.ball.velocity, context.my_state.position):
            return PositioningState
        return None

    def execute(self, context: RobotContext, **kwargs: Any) -> None:
        sender: grSimSender = kwargs["sender"]
        movement = kwargs["movement"]
        
        interception_point = calculate_interception_point(context.ball.position, context.ball.velocity)
        
        vx, vy = movement.go_to(
            context.my_state.position,
            interception_point,
            kicker_to_target=True,
            max_speed=INTERCEPT_MAX_SPEED,
        )
        w = movement.face_towards(context.my_state.position, context.ball.position)
        
        sender.send_command(
            RobotCommand(robot_id=context.robot_id, vx=vx, vy=vy, w=w, kick=0, dribble=1)
        )


class ReceivingStateMachine(StateMachine[RobotContext]):
    """State machine for receiving passes."""
    
    def __init__(self):
        positioning = PositioningState()
        intercepting = InterceptingState()
        
        super().__init__(initial_state=positioning)
        self.register_state(intercepting)
