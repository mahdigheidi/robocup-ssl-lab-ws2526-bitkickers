
# Basic implementation of mid fielder 
from enum import Enum, auto
import numpy as np

from TeamControl.robot.role import Role
from TeamControl.robot.context import RobotContext, POSSESSION_THRESHOLD
from TeamControl.utils.state_machine import StateMachine
from TeamControl.robot.phases.ball_handling import (
    ChasingBallStateMachine,
    PassingStateMachine,
)

STRIKER_ID = 3


def has_possession(ball_pos, robot_pos):
    return np.linalg.norm(ball_pos - robot_pos[:2]) < POSSESSION_THRESHOLD


class Support(Role):
    class Phase(Enum):
        CHASING = auto()
        PASSING = auto()

    def get_initial_phase(self):
        return self.Phase.CHASING

    def create_phase_machine(self, phase):
        if phase == self.Phase.CHASING:
            return ChasingBallStateMachine()
        if phase == self.Phase.PASSING:
            return PassingStateMachine()
        raise ValueError(phase)

    def update_phase(self, robot, context: RobotContext):

        # ================= CHASING =================
        if robot.current_phase == self.Phase.CHASING:
            if context.has_possession:
                # Inject ONLY striker as target
                if STRIKER_ID in context.teammates:
                    context.teammates = {
                        STRIKER_ID: context.teammates[STRIKER_ID]
                    }
                robot.switch_phase(self.Phase.PASSING)
                return

        # ================= PASSING =================
        if robot.current_phase == self.Phase.PASSING:
            if not context.has_possession:
                robot.switch_phase(self.Phase.CHASING)
                return
