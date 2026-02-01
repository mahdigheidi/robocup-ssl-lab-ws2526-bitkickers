# goalie role 

import logging
from enum import Enum, auto
from TeamControl.robot.role import Role
from TeamControl.robot.context import RobotContext
from TeamControl.utils.state_machine import StateMachine
from TeamControl.obstacles.helpers import ball_in_defense_area
from TeamControl.robot.phases.ball_handling import (
    ChasingBallStateMachine,
    PassingStateMachine,
)
from TeamControl.robot.phases.basic import GoalieStateMachine

logger = logging.getLogger(__name__)


class Goalkeeper(Role):
    class Phase(Enum):
        ACTIVE = auto()
        CHASING = auto()
        PASSING = auto()

    def get_initial_phase(self):
        return self.Phase.ACTIVE

    def create_phase_machine(self, phase: Phase) -> StateMachine[RobotContext]:
        #logger.info(f"[GOALIE] Creating phase machine: {phase.name}")

        if phase == self.Phase.ACTIVE:
            return GoalieStateMachine()

        if phase == self.Phase.CHASING:
            return ChasingBallStateMachine()

        if phase == self.Phase.PASSING:
            return PassingStateMachine()

        raise ValueError(f"Unknown phase: {phase}")
    


    def update_phase(self, robot, context: RobotContext):
        # Check if ball is inside THIS goalie's defense area
       
        if context.is_yellow:
            y_defense = ball_in_defense_area(context.ball.position)
        else:
            # ball for blue side
            b_ball = (-context.ball.position[0], context.ball.position[1])
            y_defense = ball_in_defense_area(b_ball)

        #ACTIVE 
        if robot.current_phase == self.Phase.ACTIVE:
            if y_defense:
                #logger.info(
                #    f"[GOALIE {robot.robot_id}] ACTIVE → CHASING (ball in own defense)"
                #)
                robot.switch_phase(self.Phase.CHASING)
                return

        # CHASING
        elif robot.current_phase == self.Phase.CHASING:

            if context.has_possession:
                #logger.info(
                #    f"[GOALIE {robot.robot_id}] CHASING → PASSING (has possession)"
                #)
                context.teammates = ordered_pass_targets(context)
                robot.switch_phase(self.Phase.PASSING)
                return

            if not y_defense:
                #logger.info(
                #    f"[GOALIE {robot.robot_id}] CHASING → ACTIVE (ball left defense)"
                #)
                robot.switch_phase(self.Phase.ACTIVE)
                return

        # PASSING
        elif robot.current_phase == self.Phase.PASSING:
            if not context.has_possession:
                #logger.info(
                #    f"[GOALIE {robot.robot_id}] PASSING → ACTIVE (lost possession)"
                #)
                robot.switch_phase(self.Phase.ACTIVE)
                return


# Ordered pass targets 

def ordered_pass_targets(context: RobotContext):
    """
    defender (1) → support (2) → striker (3)
    """
    PASS_PRIORITY = [1, 2, 3]
    ordered = {}

    for rid in PASS_PRIORITY:
        if rid in context.teammates:
            ordered[rid] = context.teammates[rid]

    #logger.info(f"[GOALIE] Ordered pass targets: {list(ordered.keys())}")
    return ordered
