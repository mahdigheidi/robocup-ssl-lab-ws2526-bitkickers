"""Receiver state classes - states for the robot waiting to receive the ball."""

from typing import Optional, Any, Type
import numpy as np
from TeamControl.utils.state_machine import State
from TeamControl.network.ssl_sockets import grSimSender
from TeamControl.network.robot_command import RobotCommand
from TeamControl.ball.ball_idle_spawner import BallIdleSpawner
from TeamControl.robot.robot_state_machine import RobotContext


# Interception constants
INTERCEPT_SPEED_THRESHOLD = 800  # mm/s - minimum ball speed to use interception
INTERCEPT_DOT_THRESHOLD = 0.5  # Cosine threshold for "ball coming towards me"
INTERCEPT_BACKWARD_SHIFT = 0.8  # Factor to shift interception point backward
BALL_SLOW_THRESHOLD = 500  # mm/s - below this, ball is considered stationary


def should_receiver_intercept(
    ball_pos: np.ndarray,
    ball_velocity: np.ndarray,
    receiver_pos: np.ndarray,
) -> bool:
    """
    Determine if receiver should intercept incoming ball vs positioning.

    Returns True if:
    - Ball is moving fast enough (> INTERCEPT_SPEED_THRESHOLD)
    - Ball is moving towards receiver (dot product > INTERCEPT_DOT_THRESHOLD)
    """
    ball_speed = np.linalg.norm(ball_velocity)

    if ball_speed < INTERCEPT_SPEED_THRESHOLD:
        return False

    # Check if ball is moving towards receiver
    ball_direction = ball_velocity / ball_speed
    to_receiver = receiver_pos[:2] - ball_pos[:2]
    dist_to_receiver = np.linalg.norm(to_receiver)

    if dist_to_receiver == 0:
        return False

    to_receiver_norm = to_receiver / dist_to_receiver
    dot = np.dot(ball_direction, to_receiver_norm)

    return dot > INTERCEPT_DOT_THRESHOLD


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
    """
    Calculate position for receiver robot at minimum distance from passer.
    Ensures receiver stays within field boundaries.
    """
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

    # If clamping brought us closer than min_distance, project back onto circle
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


class PositioningState(State["RobotContext"]):
    """Receiver state: maintain distance from passer and face towards passer."""

    def handle_transition(self, context: "RobotContext") -> Optional[Type[State["RobotContext"]]]:
        if context.my_role != "receiver":
            return None

        # Check if should intercept incoming ball
        should_intercept = should_receiver_intercept(
            context.ball_pos, context.ball_velocity, context.my_pos
        )

        if should_intercept:
            return InterceptingState

        return None

    def execute(self, context: "RobotContext", **kwargs: Any) -> None:
        if context.my_role != "receiver":
            return

        sender: grSimSender = kwargs["sender"]
        spawner: BallIdleSpawner = kwargs["spawner"]
        passing_min_distance: float = kwargs["passing_min_distance"]

        # Position at minimum distance from passer (other_pos)
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


class InterceptingState(State["RobotContext"]):
    """Receiver state: intercept incoming pass."""

    def handle_transition(self, context: "RobotContext") -> Optional[Type[State["RobotContext"]]]:
        if context.my_role != "receiver":
            return None

        # Check if ball is still coming towards receiver
        ball_speed = np.linalg.norm(context.ball_velocity)

        if ball_speed < BALL_SLOW_THRESHOLD:
            # Ball stopped - switch to positioning
            return PositioningState

        # Check if ball is still moving towards receiver
        ball_direction = context.ball_velocity / ball_speed
        to_receiver = context.my_pos[:2] - context.ball_pos[:2]
        dist_to_receiver = np.linalg.norm(to_receiver)

        if dist_to_receiver > 0:
            to_receiver_norm = to_receiver / dist_to_receiver
            dot = np.dot(ball_direction, to_receiver_norm)

            if dot < INTERCEPT_DOT_THRESHOLD:
                # Ball no longer coming towards us - position
                return PositioningState

        return None

    def execute(self, context: "RobotContext", **kwargs: Any) -> None:
        if context.my_role != "receiver":
            return

        sender: grSimSender = kwargs["sender"]

        # Calculate interception point using numpy operations
        ball_speed = np.linalg.norm(context.ball_velocity)

        if ball_speed > 0:
            # Vector from ball to receiver
            ball_to_receiver = context.my_pos[:2] - context.ball_pos[:2]

            # Project onto ball velocity to find closest point on trajectory
            dot_product = np.dot(ball_to_receiver, context.ball_velocity)
            velocity_squared = np.dot(context.ball_velocity, context.ball_velocity)
            t = dot_product / velocity_squared if velocity_squared > 0 else 0
            t = max(0, t)  # Don't go behind the ball

            # Closest point on trajectory
            closest_point = context.ball_pos[:2] + t * context.ball_velocity

            # Calculate distance from receiver to closest point on trajectory
            distance_to_line = np.linalg.norm(context.my_pos[:2] - closest_point)

            # Shift interception point backward to give more time
            shift_distance = distance_to_line * INTERCEPT_BACKWARD_SHIFT
            t_shift = shift_distance / ball_speed
            t = max(0, t - t_shift)

            # Final interception target
            target = context.ball_pos[:2] + t * context.ball_velocity
        else:
            target = context.ball_pos[:2]

        # Move to interception point
        vx, vy = context.my_movement.go_to(
            context.my_pos,
            target,
            kicker_to_target=True,
        )
        w = context.my_movement.face_towards(
            robot_pos=context.my_pos, target=context.ball_pos
        )
        sender.send_command(
            RobotCommand(robot_id=context.robot_id, vx=vx, vy=vy, w=w, kick=0, dribble=1)
        )
