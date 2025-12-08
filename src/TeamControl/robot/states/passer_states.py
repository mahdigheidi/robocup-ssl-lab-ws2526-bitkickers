"""Passer state classes - states for the robot responsible for getting and passing the ball."""

from typing import Optional, Any, Type
import numpy as np
from TeamControl.utils.state_machine import State
from TeamControl.network.ssl_sockets import grSimSender
from TeamControl.network.robot_command import RobotCommand
from TeamControl.robot.robot_state_machine import RobotContext


def distance(pos0: np.ndarray, pos1: np.ndarray) -> float:
    """Calculate Euclidean distance between two 2D positions."""
    return float(np.linalg.norm(pos0[:2] - pos1[:2]))


class ChasingBallState(State["RobotContext"]):
    """Passer state: chase ball from far away - fast approach."""

    def handle_transition(self, context: "RobotContext") -> Optional[Type[State["RobotContext"]]]:
        if context.my_role != "passer":
            return None

        ball_dist = distance(context.ball_pos, context.my_pos)

        # Close enough to switch to gentle approach
        if ball_dist < 200:
            return ApproachingBallState

        return None

    def execute(self, context: "RobotContext", **kwargs: Any) -> None:
        if context.my_role != "passer":
            return

        sender: grSimSender = kwargs["sender"]

        # Chase the ball quickly
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


class ApproachingBallState(State["RobotContext"]):
    """Passer state: approach ball gently when close - prevents bumping."""

    def handle_transition(self, context: "RobotContext") -> Optional[Type[State["RobotContext"]]]:
        if context.my_role != "passer":
            return None

        ball_dist = distance(context.ball_pos, context.my_pos)

        # Ball moved far away - go back to fast chasing
        if ball_dist > 250:
            return ChasingBallState

        # Got possession - start aligning
        if context.has_possession:
            return AligningWithTargetState

        return None

    def execute(self, context: "RobotContext", **kwargs: Any) -> None:
        if context.my_role != "passer":
            return

        sender: grSimSender = kwargs["sender"]

        # Approach ball gently to avoid bumping
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


class AligningWithTargetState(State["RobotContext"]):
    """Passer state: align with target (receiver) to pass."""

    def handle_transition(self, context: "RobotContext") -> Optional[Type[State["RobotContext"]]]:
        if context.my_role != "passer":
            return None

        # Lost possession - go back to approaching
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

    def execute(self, context: "RobotContext", **kwargs: Any) -> None:
        if context.my_role != "passer":
            return

        sender: grSimSender = kwargs["sender"]

        # Align with target robot (receiver), dribbler on
        vx, vy = 0, 0
        w = context.my_movement.face_towards(
            robot_pos=context.my_pos,
            target=context.other_pos,
            max_angular_acceleration=3,
        )
        sender.send_command(
            RobotCommand(robot_id=context.robot_id, vx=vx, vy=vy, w=w, kick=0, dribble=1)
        )


class KickingState(State["RobotContext"]):
    """Passer state: kick the ball."""

    def handle_transition(self, context: "RobotContext") -> Optional[Type[State["RobotContext"]]]:
        if context.my_role != "passer":
            return None

        if not context.has_possession:
            return ChasingBallState
        return None

    def execute(self, context: "RobotContext", **kwargs: Any) -> None:
        if context.my_role != "passer":
            return

        sender: grSimSender = kwargs["sender"]

        # Kick the ball
        vx, vy = 0, 0
        w = 0
        sender.send_command(
            RobotCommand(robot_id=context.robot_id, vx=vx, vy=vy, w=w, kick=5, dribble=0)
        )
