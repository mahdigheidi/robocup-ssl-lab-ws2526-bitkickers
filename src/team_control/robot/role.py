"""Robot roles - defines strategic roles and role interface."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from team_control.utils.state_machine import StateMachine
    from team_control.robot.context import RobotContext


class Role(ABC):
    """Abstract base class for robot strategic roles.

    Each role (Attacker, Goalkeeper, etc.) extends this class and defines:
    - Its own Phase enum (what tactical phases it can enter)
    - Initial phase
    - Phase-to-state-machine mapping
    """

    @abstractmethod
    def get_initial_phase(self) -> Enum:
        """Return the starting phase for this role."""
        pass

    @abstractmethod
    def create_phase_machine(self, phase: Enum) -> "StateMachine[RobotContext]":
        """Create state machine for the given phase.

        Args:
            phase: The phase to create a state machine for

        Returns:
            StateMachine instance for that phase
        """
        pass
