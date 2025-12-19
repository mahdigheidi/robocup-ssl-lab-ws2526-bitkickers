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
        
        # Calculate kick power based on distance to target
        if context.teammates:
            target_pos = list(context.teammates.values())[0].position
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

# Repositioning constants
REPOSITION_STEP_SIZE = 800  # mm - how far to move when repositioning
REPOSITION_MAX_SPEED = 2500  # mm/s - speed during repositioning
SHADOW_WIDTH = 300  # mm - how wide the "blocked" zone is around the passing lane


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
    # (distance to field boundary)
    left_candidate = receiver_pos[:2] + perp_left * REPOSITION_STEP_SIZE
    right_candidate = receiver_pos[:2] + perp_right * REPOSITION_STEP_SIZE
    
    # Check how far each candidate is from nearest boundary
    def space_available(pos: np.ndarray) -> float:
        """Return minimum distance to any field boundary."""
        return min(
            pos[0] - field_x_min,
            field_x_max - pos[0],
            pos[1] - field_y_min,
            field_y_max - pos[1],
        )
    
    left_space = space_available(left_candidate)
    right_space = space_available(right_candidate)
    
    # Choose direction with more open space
    if left_space > right_space:
        candidate = left_candidate
    else:
        candidate = right_candidate
    
    # Clamp to field boundaries
    candidate = np.array([
        max(field_x_min + margin, min(field_x_max - margin, candidate[0])),
        max(field_y_min + margin, min(field_y_max - margin, candidate[1])),
    ])
    
    return candidate


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
        
        # Get first teammate position (for two-robot passing)
        if not context.teammates:
            return
        
        passer_pos = list(context.teammates.values())[0].position
        
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
                    vx, vy = movement.go_to(context.my_state.position, target_pos, kicker_to_target=True)
                    w = movement.face_towards(context.my_state.position, context.ball.position)
                    sender.send_command(
                        RobotCommand(robot_id=context.robot_id, vx=vx, vy=vy, w=w, kick=0, dribble=1)
                    )
                    return
        
        # Get opponent state (first opponent if exists)
        opponent_state = None
        if context.opponents:
            opponent_state = list(context.opponents.values())[0]
        
        # Check if we need to reposition to avoid opponent's shadow
        target_pos = calculate_receiver_position(
            context.my_state.position,
            passer_pos,
            opponent_state,
            spawner.field_x_min,
            spawner.field_x_max,
            spawner.field_y_min,
            spawner.field_y_max,
        )
        
        if target_pos is not None:
            # Need to move to get clear
            vx, vy = movement.go_to(context.my_state.position, target_pos, max_speed=REPOSITION_MAX_SPEED)
        else:
            # Already in good position, stay still
            vx, vy = 0, 0
        
        w = movement.face_towards(context.my_state.position, passer_pos)
        
        sender.send_command(
            RobotCommand(robot_id=context.robot_id, vx=vx, vy=vy, w=w, kick=0, dribble=0)
        )


class InterceptingState(State[RobotContext]):
    """Intercept incoming ball."""

    def handle_transition(self, context: RobotContext) -> Optional[Type[State[RobotContext]]]:
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
        
        if not should_intercept(context.ball.position, context.ball.velocity, context.my_state.position):
            return PositioningState
        return None

    def execute(self, context: RobotContext, **kwargs: Any) -> None:
        sender: grSimSender = kwargs["sender"]
        movement = kwargs["movement"]
        spawner: BallIdleSpawner = kwargs["spawner"]
        
        interception_point = calculate_interception_point(
            context.ball.position, context.ball.velocity, context.my_state.position
        )
        
        # Clamp interception point to field boundaries
        margin = RECEIVER_FIELD_MARGIN
        interception_point = np.array([
            max(spawner.field_x_min + margin, min(spawner.field_x_max - margin, interception_point[0])),
            max(spawner.field_y_min + margin, min(spawner.field_y_max - margin, interception_point[1])),
        ])
        
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
