from enum import Enum, auto
from TeamControl.robot.role import Role
from TeamControl.robot.context import RobotContext
from TeamControl.utils.state_machine import StateMachine
from TeamControl.robot.phases import HoldPositionFaceBallStateMachine
from TeamControl.robot.phases.basic import GoalieStateMachine

class Goalkeeper(Role):
    class Phase(Enum):
        ACTIVE = auto()

    def get_initial_phase(self):
        return self.Phase.ACTIVE

    def create_phase_machine(self, phase):
        return GoalieStateMachine()
