from enum import Enum, auto
from TeamControl.robot.role import Role
from TeamControl.robot.context import RobotContext
from TeamControl.utils.state_machine import StateMachine
from TeamControl.robot.phases import HoldPositionFaceBallStateMachine

class Defender(Role):
    class Phase(Enum):
        HOLD = auto()

    def __init__(self, home=(2200.0, -1000.0)):
        self.home = home

    def get_initial_phase(self) -> Enum:
        return self.Phase.HOLD

    def create_phase_machine(self, phase: Enum) -> StateMachine[RobotContext]:
        return HoldPositionFaceBallStateMachine(self.home)
