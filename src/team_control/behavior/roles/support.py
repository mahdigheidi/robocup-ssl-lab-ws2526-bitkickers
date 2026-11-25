import logging
from enum import Enum, auto

from team_control.behavior.role import Role
from team_control.behavior.context import RobotContext
from team_control.utils.state_machine import StateMachine
from team_control.behavior.phases.ball_handling import (
    ChasingBallStateMachine,
    #PassingStateMachine,
)
from team_control.behavior.phases.basic import (
    HoldPositionFaceBallStateMachine,
  
    KickToGoalStateMachine   , 
)

logger = logging.getLogger(__name__)

STRIKER_ID = 3

SUPPORT_X_MIN = -1500.0 
SUPPORT_X_MAX = 1500.0 
SUPPORT_Y_MIN = -3000.0 
SUPPORT_Y_MAX = 3000.0

def inside_zone(context: RobotContext) -> bool:
    x, y = context.ball.position[:2]
    return (
        SUPPORT_X_MIN <= x <= SUPPORT_X_MAX and
        SUPPORT_Y_MIN <= y <= SUPPORT_Y_MAX
    )


class Support(Role):
    class Phase(Enum):
        CHASING = auto()
      # PASSING = auto()
        HOLD = auto()
        KICK_TO_GOAL = auto()


    # --------------------------------------------------
    # Initial phase
    # --------------------------------------------------
    def get_initial_phase(self):
        return self.Phase.CHASING

    # --------------------------------------------------
    # Phase → StateMachine mapping
    # --------------------------------------------------
    def create_phase_machine(self, phase: Phase) -> StateMachine[RobotContext]:
        #logger.info(f"[SUPPORT] Creating phase machine: {phase.name}")

        if phase == self.Phase.CHASING:
            return ChasingBallStateMachine()

        #if phase == self.Phase.PASSING:
           # return PassingStateMachine()

        if phase == self.Phase.KICK_TO_GOAL:
            return KickToGoalStateMachine()
        
        if phase == self.Phase.HOLD:
            return HoldPositionFaceBallStateMachine(home=(0.0, 0.0))
  
        raise ValueError(f"Unknown phase: {phase}")

    # --------------------------------------------------
    # Phase transitions
    def update_phase(self, robot, context: RobotContext):
        # -------- OUTSIDE ZONE CHECK --------
        inside = inside_zone(context)
        ball_inside = inside_zone(context)

        if not ball_inside:
            if robot.current_phase != self.Phase.HOLD:
               
                robot.switch_phase(self.Phase.HOLD)
            return

        if robot.current_phase == self.Phase.HOLD:

            robot.switch_phase(self.Phase.CHASING)
            return

        # -------- CHASING --------
        if robot.current_phase == self.Phase.CHASING:
            if context.has_possession:
                
                robot.switch_phase(self.Phase.KICK_TO_GOAL)
                return

        # -------- KICK TO GOAL --------
        elif robot.current_phase == self.Phase.KICK_TO_GOAL:
            if not context.has_possession:
                
                robot.switch_phase(self.Phase.CHASING)
                return