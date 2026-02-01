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


class Defender(Role):
    class Phase(Enum):
        CHASING = auto()       # Get free ball
        INTERCEPTING = auto()  # Pressure opponent
        PASSING = auto()       # Pass to teammate
        CLEARING = auto()      # Shoot upfield

    def __init__(self, home=(2200.0, -1000.0)):
        self.home = home

    def get_initial_phase(self) -> Enum:
        return self.Phase.CHASING

    def create_phase_machine(self, phase: Phase) -> StateMachine[RobotContext]:
        if phase == Defender.Phase.CHASING:
            return ChasingBallStateMachine()
        elif phase == Defender.Phase.INTERCEPTING:
            return InterceptionStateMachine()
        elif phase == Defender.Phase.PASSING:
            return PassingStateMachine()
        elif phase == Defender.Phase.CLEARING:
            return ShootingStateMachine()
