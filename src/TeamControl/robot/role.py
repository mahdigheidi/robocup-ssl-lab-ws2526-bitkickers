"""Robot roles - defines strategic roles and role interface."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, TypeVar, Generic

if TYPE_CHECKING:
    from TeamControl.utils.state_machine import StateMachine
    from TeamControl.robot.context import RobotContext

PhaseT = TypeVar("PhaseT", bound=Enum)


class Role(ABC, Generic[PhaseT]):
    """Abstract base class for robot strategic roles.
    
    Each role (Attacker, Goalkeeper, etc.) extends this class and defines:
    - Its own Phase enum (what tactical phases it can enter)
    - Initial phase
    - Phase-to-state-machine mapping
    """

    @abstractmethod
    def get_initial_phase(self) -> PhaseT:
        """Return the starting phase for this role."""
        pass

    @abstractmethod
    def create_phase_machine(self, phase: PhaseT) -> "StateMachine[RobotContext]":
        """Create state machine for the given phase.
        
        Args:
            phase: The phase to create a state machine for
            
        Returns:
            StateMachine instance for that phase
        """
        pass
