"""Passing role - for two-robot passing exercise."""
from enum import Enum, auto
from TeamControl.robot.role import Role
from TeamControl.robot.context import RobotContext
from TeamControl.utils.state_machine import StateMachine
from TeamControl.robot.phases import (
    ChasingBallStateMachine,
    PassingStateMachine,
    ReceivingStateMachine,
)



class Passer(Role):
    """Role for two-robot passing drill.
    
    Both robots use this role and dynamically switch between
    ball-handling phases based on ball responsibility.
    """

    class Phase(Enum):
        """Phases for passing."""
        CHASING = auto()      # Chase and capture ball
        PASSING = auto()      # Align and kick to other robot
        RECEIVING = auto()    # Position and intercept passes
    
    def get_initial_phase(self) -> Enum:
        """Start in receiving phase (waiting for ball)."""
        return self.Phase.RECEIVING
    
    def create_phase_machine(self, phase: Enum) -> StateMachine[RobotContext]:
        """Create state machine for given phase."""
        if phase == self.Phase.CHASING:
            return ChasingBallStateMachine()
        elif phase == self.Phase.PASSING:
            return PassingStateMachine()
        elif phase == self.Phase.RECEIVING:
            return ReceivingStateMachine()
        else:
            raise ValueError(f"Unknown phase: {phase}")
 