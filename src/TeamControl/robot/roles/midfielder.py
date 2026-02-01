from enum import Enum, auto
from TeamControl.robot.role import Role
from TeamControl.robot.context import RobotContext
from TeamControl.utils.state_machine import StateMachine
from TeamControl.robot.phases import (
    ChasingBallStateMachine,
    InterceptionStateMachine,
    PassingStateMachine,
    ShootingStateMachine,
)


class Midfielder(Role):
    class Phase(Enum):
        CHASING = auto()       # Get free ball
        INTERCEPTING = auto()  # Pressure opponent
        PASSING = auto()       # Pass to striker
        SHOOTING = auto()      # Shoot at goal

    def __init__(self, home=(0.0, 0.0)):
        self.home = home

    def get_initial_phase(self) -> Enum:
        return self.Phase.CHASING

    def create_phase_machine(self, phase: Phase) -> StateMachine[RobotContext]:
        if phase == Midfielder.Phase.CHASING:
            return ChasingBallStateMachine()
        elif phase == Midfielder.Phase.INTERCEPTING:
            return InterceptionStateMachine()
        elif phase == Midfielder.Phase.PASSING:
            return PassingStateMachine()
        elif phase == Midfielder.Phase.SHOOTING:
            return ShootingStateMachine()
