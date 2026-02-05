from __future__ import annotations
import logging
from typing import Optional
from enum import Enum, auto
from TeamControl.obstacles.helpers import RoleArea, check_possession
from TeamControl.robot import robot
from TeamControl.robot.phases.ball_handling import ReceivingStateMachine
from TeamControl.robot.phases.basic import ShadowBallStateMachine
from TeamControl.robot.role import Role
from TeamControl.robot.context import RobotContext, teammate_can_chase_ball
from TeamControl.utils.state_machine import StateMachine
from TeamControl.robot.phases import (
    ChasingBallStateMachine,
    InterceptionStateMachine,
    BlockingStateMachine,
    PassingStateMachine,
    ShootingStateMachine,
)
import numpy as np
import time

DEFENDER_AREA = RoleArea(
    x_min=1500.0,
    x_max=4500.0,
    y_min=-2900.0,
    y_max=2900.0,
    defense_x=3450.0,      # goal box edge
    defense_y_half=1050.0,
)
ROLE_AREA = DEFENDER_AREA

logger = logging.getLogger(__name__)




class Defender(Role):
    class Phase(Enum):
        CHASING = auto()       # Get free ball
        INTERCEPTING = auto()  # Pressure opponent
        BLOCKING = auto()      # Stay between opponent and goal
        PASSING = auto()       # Pass to teammate
        CLEARING = auto()  # Shoot upfield
        RECEIVING = auto()     # Receive pass
        SHADOW = auto()      # Shadow ball outside area
 

    def __init__(self, home=(2200.0, -1000.0)):
        super().__init__()
        self.home = home
        self._possession_start = None

    def get_initial_phase(self) -> Enum:
        return self.Phase.CHASING

    def create_phase_machine(self, phase: Phase) -> StateMachine[RobotContext]:
        if phase == Defender.Phase.CHASING:
            return ChasingBallStateMachine(ROLE_AREA)
        elif phase == Defender.Phase.INTERCEPTING:
            return InterceptionStateMachine(ROLE_AREA)
        elif phase == Defender.Phase.BLOCKING:
            return BlockingStateMachine(ROLE_AREA)
        elif phase == Defender.Phase.PASSING:
            return PassingStateMachine(ROLE_AREA)
        elif phase == Defender.Phase.CLEARING:
            return ShootingStateMachine(ROLE_AREA)
        
        elif phase == Defender.Phase.RECEIVING:
            return ReceivingStateMachine(ROLE_AREA)
        
        elif phase == Defender.Phase.SHADOW:
            return ShadowBallStateMachine(ROLE_AREA)

    def update_phase(self, robot, context: RobotContext):

        bx, by = context.ball.position[:2]
        ball_in_zone = DEFENDER_AREA.is_point_legal(context, bx, by)
        ball_speed = np.linalg.norm(context.ball.velocity)

        # --------------------------------------------------
        # Conflict Resolution: Multiple robots with ball
        # --------------------------------------------------
        my_dist = np.linalg.norm(context.my_state.position[:2] - context.ball.position[:2])
        is_near_ball = my_dist < 180.0

        if is_near_ball:
            if self._possession_start is None:
                self._possession_start = time.time()
            
            # Check for ANY teammate also near the ball
            conflicting_teammates = []
            for t_id, t in context.teammates.items():
                t_dist = np.linalg.norm(t.position[:2] - context.ball.position[:2])
                if t_dist < 200.0: # teammate is also near/contesting
                    conflicting_teammates.append(t_id)

            if conflicting_teammates:
                # If conflict lasts > 1.5s, robot with higher ID leaves
                if time.time() - self._possession_start > 1.5:
                    if any(t_id < context.robot_id for t_id in conflicting_teammates):
                        if robot.current_phase != self.Phase.INTERCEPTING:
                            robot.switch_phase(self.Phase.INTERCEPTING)
                        return
        else:
            if not context.has_possession:
                self._possession_start = None

        # --------------------------------------------------
        # 1) Defender has the ball → PASSING or CLEARING
        # --------------------------------------------------
        if context.has_possession:
            # Fallback: if held for > 3s in any state, ensure we are in CLEARING
            if self._possession_start is not None and (time.time() - self._possession_start > 2.0):
                if robot.current_phase != self.Phase.CLEARING:
                    robot.switch_phase(self.Phase.CLEARING)
                return

            if robot.current_phase not in [self.Phase.PASSING, self.Phase.CLEARING]:
                best_pass = select_best_pass_target(context)
                if best_pass is not None:
                    context.preferred_teammate = best_pass
                    robot.switch_phase(self.Phase.PASSING)
                else:
                    # Fallback to clearing/shooting if no good pass targets
                    robot.switch_phase(self.Phase.CLEARING)
            return

        # --------------------------------------------------
        # 2) Teammate has the ball → RECEIVE/INTERCEPT
        # --------------------------------------------------
        teammate_has_ball = any(
            check_possession(context.ball.position, t.position)
            for t in context.teammates.values()
        )
        if teammate_has_ball:
            # Switch to INTERCEPTING if teammate has ball, to block lanes
            # Or RECEIVING if we are waiting for a pass
            if robot.current_phase != self.Phase.INTERCEPTING:
                #logger.info("Defender: teammate has ball → INTERCEPTING")
                robot.switch_phase(self.Phase.INTERCEPTING)
            return

        # --------------------------------------------------
        # 3) Opponent has the ball OR fast ball → INTERCEPT
        # --------------------------------------------------
        opponent_has_ball = any(
            check_possession(context.ball.position, t.position)
            for t in context.opponents.values()
        )

        if opponent_has_ball:
            if robot.current_phase != self.Phase.BLOCKING:
                robot.switch_phase(self.Phase.BLOCKING)
            return

        if ball_speed > 500.0:
            if robot.current_phase != self.Phase.INTERCEPTING:
                robot.switch_phase(self.Phase.INTERCEPTING)
            return

        # --------------------------------------------------
        # 4) Ball inside zone & free → CHASE (only if closest)
        # --------------------------------------------------
        if ball_in_zone:
            # Tie-breaker "closest to ball" check to avoid fighting
            my_dist = np.linalg.norm(context.my_state.position[:2] - context.ball.position[:2])
            closest = True
            for t_id, t in context.teammates.items():
                # Only compare against teammates whose zone also contains the ball
                if not teammate_can_chase_ball(t_id, bx, by, context.is_yellow):
                    continue
                t_dist = np.linalg.norm(t.position[:2] - context.ball.position[:2])
                if t_dist < my_dist or (abs(t_dist - my_dist) < 2.0 and t_id < context.robot_id):
                    closest = False
                    break
            
            if closest:
                if robot.current_phase != self.Phase.CHASING:
                    #logger.info("Defender: ball free in zone and closest → CHASING")
                    robot.switch_phase(self.Phase.CHASING)
                return
            else:
                # If not closest, INTERCEPT/BLOCK instead of CHASING
                if robot.current_phase != self.Phase.INTERCEPTING:
                    #logger.info("Defender: ball free but teammate closer → INTERCEPTING")
                    robot.switch_phase(self.Phase.INTERCEPTING)
                return

        # --------------------------------------------------
        # 5) Ball outside zone → SHADOW
        # --------------------------------------------------
        if robot.current_phase != self.Phase.SHADOW:
            #logger.info("Defender: ball outside zone → SHADOW")
            robot.switch_phase(self.Phase.SHADOW)


def select_best_pass_target(context: RobotContext) -> int | None:
    """
    Select the best teammate ID to pass to.
    Prefer teammates at a reasonable passing distance.
    If everyone is too far (> 4000mm), return None to fallback to clearing.
    """
    my_pos = context.my_state.position[:2]
    best_tid = None
    max_dist = -1.0

    for tid, teammate in context.teammates.items():
        dist = np.linalg.norm(teammate.position[:2] - my_pos)
        
        # Avoid passing to robots too close (within 1000mm) or too far (> 4000mm)
        if dist < 400.0:
            continue
            
        if dist > max_dist:
            max_dist = dist
            best_tid = tid

    return best_tid
