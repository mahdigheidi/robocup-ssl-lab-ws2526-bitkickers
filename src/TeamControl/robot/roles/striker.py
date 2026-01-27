from enum import Enum, auto
from TeamControl.robot.role import Role
from TeamControl.robot.context import RobotContext
from TeamControl.utils.state_machine import StateMachine
from TeamControl.robot.phases import StrikerBasicStateMachine

class Striker(Role):
    class Phase(Enum):
        BASIC = auto()

    def __init__(self, blue_goal=( -4500.0, 0.0 )):
        self.blue_goal = blue_goal

    def get_initial_phase(self) -> Enum:
        return self.Phase.BASIC

    def create_phase_machine(self, phase: Enum) -> StateMachine[RobotContext]:
        return StrikerBasicStateMachine(self.blue_goal)
