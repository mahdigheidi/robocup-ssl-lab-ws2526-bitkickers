from enum import Enum, auto
from TeamControl.robot.role import Role
from TeamControl.robot.context import RobotContext
from TeamControl.utils.state_machine import StateMachine
from TeamControl.robot.phases import SupportBehindBallStateMachine

class Support(Role):
    class Phase(Enum):
        BASIC = auto()

    def __init__(self, yellow_goal=(4500.0, 0.0)):
        self.yellow_goal = yellow_goal

    def get_initial_phase(self) -> Enum:
        return self.Phase.BASIC

    def create_phase_machine(self, phase: Enum) -> StateMachine[RobotContext]:
        return SupportBehindBallStateMachine(self.yellow_goal)
