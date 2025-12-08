"""Shared ball handling state machines - reusable across roles."""

import numpy as np
from typing import Optional, Type, Any
from TeamControl.utils.state_machine import State, StateMachine
from TeamControl.network.ssl_sockets import grSimSender
from TeamControl.network.robot_command import RobotCommand
from TeamControl.robot.context import RobotContext
from TeamControl.ball.ball_idle_spawner import BallIdleSpawner


def distance(pos0: np.ndarray, pos1: np.ndarray) -> float:
    """Calculate Euclidean distance between two 2D positions."""
    return float(np.linalg.norm(pos0[:2] - pos1[:2]))


# =============================================================================
# Chasing Ball States
# =============================================================================

class ChasingBallState(State[RobotContext]):
    """Fast approach to ball from distance."""

    def handle_transition(self, context: RobotContext) -> Optional[Type[State[RobotContext]]]:
        ball_dist = distance(context.ball_pos, context.my_pos)
        if ball_dist < 200:
            return ApproachingBallState
        return None

    def execute(self, context: RobotContext, **kwargs: Any) -> None:
        sender: grSimSender = kwargs["sender"]
        
        vx, vy = context.my_movement.go_to(
            context.my_pos,
            context.ball_pos,
            kicker_to_target=True,
            max_acceleration=1300,
            max_speed=5000,
        )
        w = context.my_movement.face_towards(
            robot_pos=context.my_pos, target=context.ball_pos
        )
        sender.send_command(
            RobotCommand(robot_id=context.robot_id, vx=vx, vy=vy, w=w, kick=0, dribble=1)
        )


class ApproachingBallState(State[RobotContext]):
    """Gentle approach when close to ball - prevents bumping."""

    def handle_transition(self, context: RobotContext) -> Optional[Type[State[RobotContext]]]:
        ball_dist = distance(context.ball_pos, context.my_pos)
        
        if ball_dist > 250:
            return ChasingBallState
        if context.has_possession:
            return AligningWithTargetState
        return None

    def execute(self, context: RobotContext, **kwargs: Any) -> None:
        sender: grSimSender = kwargs["sender"]
        
        vx, vy = context.my_movement.go_to(
            context.my_pos,
            context.ball_pos,
            kicker_to_target=True,
            max_acceleration=800,
            max_speed=1500,
        )
        w = context.my_movement.face_towards(
            robot_pos=context.my_pos, target=context.ball_pos
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
        
        w = context.my_movement.face_towards(
            robot_pos=context.my_pos,
            target=context.other_pos,
            max_angular_acceleration=3,
        )
        if w == 0:
            return KickingState
        return None

    def execute(self, context: RobotContext, **kwargs: Any) -> None:
        sender: grSimSender = kwargs["sender"]
        
        vx, vy = 0, 0
        w = context.my_movement.face_towards(
            robot_pos=context.my_pos,
            target=context.other_pos,
            max_angular_acceleration=3,
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
            RobotCommand(robot_id=context.robot_id, vx=0, vy=0, w=0, kick=5, dribble=0)
        )


class PassingStateMachine(StateMachine[RobotContext]):
    """State machine for passing the ball to other_pos target."""
    
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
    
    if ball_speed < 500:  # Ball too slow, just position
        return False
    
    if ball_speed < 800:  # Moving but not fast
        return False
    
    # Check if ball moving toward robot
    ball_direction = ball_velocity / ball_speed
    to_robot = robot_pos[:2] - ball_pos
    to_robot_dist = np.linalg.norm(to_robot)
    
    if to_robot_dist == 0:
        return False
    
    to_robot_normalized = to_robot / to_robot_dist
    dot = np.dot(ball_direction, to_robot_normalized)
    
    return dot > 0.5  # Moving toward robot


def calculate_interception_point(ball_pos: np.ndarray, ball_velocity: np.ndarray) -> np.ndarray:
    """Calculate where to position to intercept ball."""
    ball_speed = np.linalg.norm(ball_velocity)
    if ball_speed == 0:
        return ball_pos
    
    # Project backward along trajectory
    ball_direction = ball_velocity / ball_speed
    shift_distance = 0.8 * ball_speed
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
    margin: float = 200,
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
        if should_intercept(context.ball_pos, context.ball_velocity, context.my_pos):
            return InterceptingState
        return None

    def execute(self, context: RobotContext, **kwargs: Any) -> None:
        sender: grSimSender = kwargs["sender"]
        spawner: BallIdleSpawner = kwargs["spawner"]
        passing_min_distance: float = kwargs.get("passing_min_distance", 3000)
        
        target_pos = calculate_receiver_position(
            context.other_pos,
            context.my_pos,
            passing_min_distance,
            spawner.field_x_min,
            spawner.field_x_max,
            spawner.field_y_min,
            spawner.field_y_max,
        )
        
        vx, vy = context.my_movement.go_to(context.my_pos, target_pos)
        w = context.my_movement.face_towards(context.my_pos, context.other_pos)
        
        sender.send_command(
            RobotCommand(robot_id=context.robot_id, vx=vx, vy=vy, w=w, kick=0, dribble=0)
        )


class InterceptingState(State[RobotContext]):
    """Intercept incoming ball."""

    def handle_transition(self, context: RobotContext) -> Optional[Type[State[RobotContext]]]:
        if not should_intercept(context.ball_pos, context.ball_velocity, context.my_pos):
            return PositioningState
        return None

    def execute(self, context: RobotContext, **kwargs: Any) -> None:
        sender: grSimSender = kwargs["sender"]
        
        interception_point = calculate_interception_point(context.ball_pos, context.ball_velocity)
        
        vx, vy = context.my_movement.go_to(
            context.my_pos,
            interception_point,
            kicker_to_target=True,
            max_speed=3000,
        )
        w = context.my_movement.face_towards(context.my_pos, context.ball_pos)
        
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
