"""Robot state machine - manages state transitions for a single robot based on its role."""

import time
from typing import Optional
import numpy as np
from TeamControl.utils.state_machine import State, StateMachine
from TeamControl.robot.Movement import RobotMovement
from TeamControl.robot.states import (
    ChasingBallState,
    ApproachingBallState,
    AligningWithTargetState,
    KickingState,
    PositioningState,
    InterceptingState,
)
from dataclasses import dataclass


@dataclass
class RobotContext:
    """Context for a single robot's decision making."""

    robot_id: int
    my_role: str  # "passer" or "receiver"
    my_pos: np.ndarray
    my_movement: RobotMovement
    other_pos: np.ndarray
    ball_pos: np.ndarray
    ball_velocity: np.ndarray
    has_possession: bool


def check_possession(ball_pos: np.ndarray, robot_pos: np.ndarray, threshold: float = 120) -> bool:
    """Check if robot has possession of the ball based on distance threshold."""
    dist = float(np.linalg.norm(ball_pos[:2] - robot_pos[:2]))
    return dist != 0 and dist < threshold


class RobotStateMachine:
    """State machine for a single robot with role-based behavior."""

    def __init__(self, initial_role: str):
        """
        Initialize state machine for a robot.

        Args:
            initial_role: "passer" or "receiver"
        """
        self.current_role = initial_role

        # Create all states
        chasing = ChasingBallState()
        approaching = ApproachingBallState()
        aligning = AligningWithTargetState()
        kicking = KickingState()
        positioning = PositioningState()
        intercepting = InterceptingState()

        # Create state machine - start with role-appropriate state
        if initial_role == "passer":
            self._machine = StateMachine[RobotContext](chasing)
        else:
            self._machine = StateMachine[RobotContext](positioning)

        # Register all states
        self._machine.register_state(chasing)
        self._machine.register_state(approaching)
        self._machine.register_state(aligning)
        self._machine.register_state(kicking)
        self._machine.register_state(positioning)
        self._machine.register_state(intercepting)

    def update(
        self,
        robot_id: int,
        my_role: str,
        my_pos: np.ndarray,
        my_movement: RobotMovement,
        other_pos: np.ndarray,
        ball_pos: np.ndarray,
        ball_velocity: np.ndarray,
    ):
        """
        Update state machine based on current conditions.

        Args:
            robot_id: This robot's ID
            my_role: "passer" or "receiver"
            my_pos: This robot's position
            my_movement: This robot's movement controller
            other_pos: Other robot's position
            ball_pos: Ball position
            ball_velocity: Ball velocity
        """
        # Handle role switching
        if self.current_role != my_role:
            from TeamControl.utils.state_machine import _START_TIME

            elapsed = time.time() - _START_TIME
            print(
                f"[ROBOT {robot_id} @ {elapsed:.3f}s] Role switched: {self.current_role} -> {my_role}"
            )
            self.current_role = my_role

            # Reset to appropriate starting state
            context = RobotContext(
                robot_id=robot_id,
                my_role=my_role,
                my_pos=my_pos,
                my_movement=my_movement,
                other_pos=other_pos,
                ball_pos=ball_pos,
                ball_velocity=ball_velocity,
                has_possession=check_possession(ball_pos, my_pos),
            )

            if my_role == "passer":
                self._machine.force_transition(ChasingBallState, context)
            else:
                self._machine.force_transition(PositioningState, context)

        # Check possession
        has_possession = check_possession(ball_pos, my_pos)

        # Build context
        context = RobotContext(
            robot_id=robot_id,
            my_role=my_role,
            my_pos=my_pos,
            my_movement=my_movement,
            other_pos=other_pos,
            ball_pos=ball_pos,
            ball_velocity=ball_velocity,
            has_possession=has_possession,
        )

        # Update state machine
        self._machine.update(context)

    def execute(self, context: RobotContext, **kwargs):
        """Execute behavior for the current state."""
        self._machine.execute(context, **kwargs)
