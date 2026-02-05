# goalie role 
from typing import Optional
import logging
import time
import numpy as np
from enum import Enum, auto
from TeamControl.robot.role import Role
from TeamControl.robot.context import RobotContext
from TeamControl.utils.state_machine import StateMachine
from TeamControl.robot.phases.ball_handling import (
    ChasingBallStateMachine,
    PassingStateMachine,
)
from TeamControl.robot.phases import (
    GoalieStateMachine, 
    ShootingStateMachine,
)
from TeamControl.obstacles.helpers import RoleArea, ball_in_defense_area, check_possession

logger = logging.getLogger(__name__)

GOALIE_AREA = RoleArea(
    x_min=3150.0,
    x_max=4500.0,
    y_min=-1200.0,
    y_max=1200.0,
)

ROLE_AREA = GOALIE_AREA

class Goalkeeper(Role):
    class Phase(Enum):
        ACTIVE = auto()
        CHASING = auto()
        PASSING = auto()
        SHOOTING = auto()

    def __init__(self):
        super().__init__()
        self._possession_start = None

    def get_initial_phase(self):
        return self.Phase.ACTIVE

    def create_phase_machine(self, phase: Phase) -> StateMachine[RobotContext]:
        #logger.info(f"[GOALIE] Creating phase machine: {phase.name}")

        if phase == self.Phase.ACTIVE:
            return GoalieStateMachine(ROLE_AREA)

        if phase == self.Phase.CHASING:
            return ChasingBallStateMachine(ROLE_AREA, is_goalie=True)

        if phase == self.Phase.PASSING:
            return PassingStateMachine(ROLE_AREA, is_goalie=True)
        if phase == self.Phase.SHOOTING:
            return ShootingStateMachine(ROLE_AREA, is_goalie=True)

        raise ValueError(f"Unknown phase: {phase}")
    

    def update_phase(self, robot, context: RobotContext):
        # Check if ball is inside THIS goalie's defense area
       
        y_defense = ball_in_defense_area(context, context.ball.position)
        teammate_has_ball = any(
            check_possession(context.ball.position, t.position)
            for t in context.teammates.values()
        )

        # Conflict Resolution: Multiple robots with ball
        my_dist = np.linalg.norm(context.my_state.position[:2] - context.ball.position[:2])
        is_near_ball = my_dist < 100.0

        if is_near_ball:
            if self._possession_start is None:
                self._possession_start = time.time()
            
            conflicting_teammates = []
            for t_id, t in context.teammates.items():
                t_dist = np.linalg.norm(t.position[:2] - context.ball.position[:2])
                if t_dist < 100.0:
                    conflicting_teammates.append(t_id)

            if conflicting_teammates:
                if time.time() - self._possession_start > 1.5:
                    if any(t_id < context.robot_id for t_id in conflicting_teammates):
                        robot.switch_phase(self.Phase.ACTIVE)
                        return
        else:
            if not context.has_possession:
                self._possession_start = None
        
        #ACTIVE 
        if robot.current_phase == self.Phase.ACTIVE:
            if y_defense:
                robot.switch_phase(self.Phase.CHASING)
                return

        # CHASING
        elif robot.current_phase == self.Phase.CHASING:
            if context.has_possession:
                self._possession_start = time.time()
                context.preferred_teammate = select_best_pass_target(context)
                robot.switch_phase(self.Phase.PASSING)
                return

            if not y_defense:
                robot.switch_phase(self.Phase.ACTIVE)
                return

        # PASSING
        elif robot.current_phase == self.Phase.PASSING:
            if not context.has_possession:
                robot.switch_phase(self.Phase.ACTIVE)
                return
            
            # Check for 2 second timeout to switch to SHOOTING
            if self._possession_start is not None and (time.time() - self._possession_start > 2.0):
                robot.switch_phase(self.Phase.SHOOTING)
                return

        # SHOOTING
        elif robot.current_phase == self.Phase.SHOOTING:
            if not context.has_possession:
                robot.switch_phase(self.Phase.ACTIVE)
                return


# Ordered pass targets 

def select_best_pass_target(context: RobotContext) -> Optional[int]:
    """
    Select the best teammate ID to pass to.
    Goalie should prefer furthest teammates (striker/midfielder) to clear the ball.
    """
    my_pos = context.my_state.position[:2]
    best_tid = None
    max_dist = -1.0

    for tid, teammate in context.teammates.items():
        dist = np.linalg.norm(teammate.position[:2] - my_pos)
        
        # Don't pass to defender if they are standing right next to us
        if dist < 400.0:
            continue
            
        if dist > max_dist:
            max_dist = dist
            best_tid = tid

    # Fallback to furthest
    if best_tid is None and context.teammates:
        best_tid = max(context.teammates.keys(), 
                       key=lambda tid: np.linalg.norm(context.teammates[tid].position[:2] - my_pos))

    return best_tid
