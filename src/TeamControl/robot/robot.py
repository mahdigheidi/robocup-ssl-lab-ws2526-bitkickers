"""Robot class - encapsulates a single robot with its role and phase-based behavior."""

import logging
from typing import Generic, TypeVar
from enum import Enum
import numpy as np
from TeamControl.robot.Movement import RobotMovement
from TeamControl.robot.role import Role
from TeamControl.robot.context import RobotContext, check_possession
from TeamControl.robot.spawn_robot import send_robot_to_grsim
from TeamControl.network.ssl_sockets import grSimSender
from TeamControl.utils.state_machine import StateMachine

R = TypeVar("R", bound=Role)

logger = logging.getLogger(__name__)


class Robot(Generic[R]):
    """
    A robot with autonomous behavior based on roles and phases.
    
    Generic over role type R for type safety (e.g., Robot[Attacker], Robot[Goalkeeper]).
    """

    def __init__(self, robot_id: int, is_yellow: bool, role: R):
        """
        Initialize a robot with a strategic role.

        Args:
            robot_id: Robot ID (0-15)
            is_yellow: True if yellow team, False if blue team
            role: Role instance defining strategic behavior
        """
        self.robot_id = robot_id
        self.is_yellow = is_yellow
        self.movement = RobotMovement()
        self.role = role

        # Current state
        self.position: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.current_phase: Enum = role.get_initial_phase()
        self._phase_machine: StateMachine[RobotContext] = role.create_phase_machine(
            self.current_phase
        )

    def spawn(self, x: float, y: float, orientation: float = 0.0):
        """
        Spawn robot in simulator at given position.

        Args:
            x: X position in mm
            y: Y position in mm
            orientation: Orientation in radians
        """
        send_robot_to_grsim(
            team_yellow=self.is_yellow, robot_id=self.robot_id, x=x, y=y, orientation=orientation
        )

    def update_position(self, position: np.ndarray):
        """Update robot's current position from world model."""
        self.position = position

    def switch_phase(self, new_phase: Enum):
        """
        Switch to a new tactical phase.

        Args:
            new_phase: Phase enum value from the role's Phase enum
        """
        if new_phase != self.current_phase:
            old_phase = self.current_phase
            self.current_phase = new_phase
            self._phase_machine = self.role.create_phase_machine(new_phase)
            logger.info(
                f"R{self.robot_id}[{self.role.__class__.__name__}] "
                f"PHASE: {old_phase.name} -> {new_phase.name}"
            )
    
    @property
    def current_state_name(self) -> str:
        """Get the name of the current state in the active phase machine."""
        return self._phase_machine.get_current_state().__class__.__name__

    def execute_behavior(
        self,
        sender: grSimSender,
        other_pos: np.ndarray,
        ball_pos: np.ndarray,
        ball_velocity: np.ndarray,
        **kwargs,
    ):
        """
        Execute the robot's current phase behavior.

        Args:
            sender: grSim command sender
            other_pos: Position of other robot(s)
            ball_pos: Ball position
            ball_velocity: Ball velocity
            **kwargs: Additional parameters needed by state machines
        """
        # Build context for execution
        context = RobotContext(
            robot_id=self.robot_id,
            my_pos=self.position,
            my_movement=self.movement,
            other_pos=other_pos,
            ball_pos=ball_pos,
            ball_velocity=ball_velocity,
            has_possession=check_possession(ball_pos, self.position),
        )

        # Execute current phase's state machine
        self._phase_machine.execute(context, sender=sender, **kwargs)