#stiker role

from enum import Enum, auto
import time
import logging
from TeamControl.obstacles.helpers import RoleArea, check_possession

from TeamControl.robot import context
from TeamControl.robot.phases.ball_handling import is_path_blocked
from TeamControl.robot.roles import goalie
import numpy as np


from TeamControl.robot.role import Role
from TeamControl.robot.context import RobotContext
from TeamControl.utils.state_machine import StateMachine

from TeamControl.robot.phases.basic import (
    HoldPositionFaceBallStateMachine,
    InterceptionStateMachine,
    ShadowBallStateMachine
)


from TeamControl.robot.phases import (
    ReceivingStateMachine,
    ChasingBallStateMachine,
    StrikerStateMachine,
    KickToGoalStateMachine,
)


BALL_SLOW_THRESHOLD = 500
BALL_DIRECTION_DOT_THRESHOLD = 0.3
GOALIE_HOLD_TIME = 1.0

STRIKER_AREA = RoleArea(
    x_min=-4400.0,
    x_max=-1400.0,
    y_min=-2900.0,
    y_max=2900.0,
    defense_x=-3400.0,      # goal box edge
    defense_y_half=1100.0,
)


logger = logging.getLogger(__name__)

def log_phase(robot, phase, reason: str):
    logger.info(
        f"[R{robot.robot_id} | {'Y' if robot.is_yellow else 'B'} | STRIKER] "
        f"→ {phase.name} | {reason}"
    )


# ============================================================================
# Striker Role
# ============================================================================
class Striker(Role):

    class Phase(Enum):
        HOLD = auto()
        SHADOW = auto()
        RECEIVING = auto()
        PLAY = auto()
        CHASING = auto()
        INTERCEPT = auto()
        KICK_TO_GOAL = auto()

    def __init__(self, blue_goal= None):
        super().__init__()
        self._possession_start = None
        self._goalie_hold_start = None
        self.blue_goal = blue_goal

    def get_initial_phase(self):
        return self.Phase.CHASING

    
    def create_phase_machine(self, phase):

        if phase == self.Phase.RECEIVING:
            return ReceivingStateMachine()

        if phase == self.Phase.PLAY:
            return StrikerStateMachine()

        if phase == self.Phase.CHASING:
            return ChasingBallStateMachine()

        if phase == self.Phase.HOLD:
            return HoldPositionFaceBallStateMachine(self.home)

        if phase == self.Phase.INTERCEPT:
            return InterceptionStateMachine(STRIKER_AREA)
        
        if phase == self.Phase.KICK_TO_GOAL:
            return KickToGoalStateMachine()
        if phase == self.Phase.SHADOW:
            return ShadowBallStateMachine(STRIKER_AREA)

        raise ValueError(phase)
    


    def opp_goalie_has_ball(self, context: RobotContext) -> bool:
        if not context.opponents:
            return False

        goal_op = -4500.0 if context.is_yellow else 4500.0

        goalie = min(
            context.opponents.values(),
            key=lambda o: abs(o.position[0] - goal_op)
        )

        return check_possession(context.ball.position, goalie.position)



# ============================================================================
# Phase Update Logic
# ============================================================================
    def update_phase(self, robot, context: RobotContext):
        
        bx, by = context.ball.position[:2]
        ball_legal =  STRIKER_AREA.is_point_legal(context, bx, by)
        

        # 5) Default → CHASING
        if ball_legal:

            if robot.current_phase != self.Phase.CHASING:
                logger.info("Striker: default → CHASING")
                robot.switch_phase(self.Phase.CHASING)
        
        # 1) We have the ball → PLAY
        if context.has_possession:
            if robot.current_phase != self.Phase.KICK_TO_GOAL:
                logger.info("Striker: has ball → KICK_TO_GOAL")
                robot.switch_phase(self.Phase.KICK_TO_GOAL)
            return
         

        # 4) Teammate has ball → RECEIVING
        if any(check_possession(context.ball.position, t.position)
               for t in context.teammates.values()):
            
            if robot.current_phase != self.Phase.RECEIVING:
                logger.info("Striker: teammate has ball → RECEIVING")

                robot.switch_phase(self.Phase.RECEIVING)
            return
        
        # 2) Opponent goalie has possession → INTERCEPT
        if self.opp_goalie_has_ball(context):

            if robot.current_phase != self.Phase.INTERCEPT:
                logger.info("Striker: opponent goalie has ball → INTERCEPT")

                robot.switch_phase(self.Phase.INTERCEPT)
            return



         # stable_has_ball = self.stable_has_ball(context)
        if not ball_legal:
            if robot.current_phase != self.Phase.SHADOW:

                logger.info("Striker: ball illegal → SHADOW")

            robot.switch_phase(self.Phase.SHADOW)
            return
   