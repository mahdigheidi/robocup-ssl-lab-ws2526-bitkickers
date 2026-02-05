import logging
from typing import Optional
from enum import Enum, auto
from TeamControl.robot.phases.ball_handling import ReceivingStateMachine
from TeamControl.robot.role import Role
from TeamControl.robot.context import RobotContext, teammate_can_chase_ball
from TeamControl.obstacles.helpers import RoleArea, check_possession
from TeamControl.robot.phases.basic import ShadowBallStateMachine
from TeamControl.utils.state_machine import StateMachine
from TeamControl.robot.phases import (
    ChasingBallStateMachine,
    InterceptionStateMachine,
    PassingStateMachine,
    ShootingStateMachine,

)
from TeamControl.robot.phases.ball_handling import is_path_clear
import numpy as np
import time

MIDFIELDER_AREA = RoleArea(
    x_min=-1500.0,
    x_max=1500.0,
    y_min=-2900.0,
    y_max=2900.0,
    #defense_x=-3400.0,
    #defense_y_half=1100.0,
)

ROLE_AREA = MIDFIELDER_AREA

logger = logging.getLogger(__name__)


class Midfielder(Role):
    class Phase(Enum):
        CHASING = auto()       # Get free ball
        INTERCEPTING = auto()  # Pressure opponent
        PASSING = auto()       # Pass to striker
        SHOOTING = auto()      # Shoot at goal
        RECEIVING = auto()     # Receive pass
        SHADOW = auto()      # Shadow ball outside area

    def __init__(self, home=(0.0, 0.0)):
        super().__init__()
        self.home = home
        self._possession_start = None
        self._force_center = False

    def get_initial_phase(self) -> Enum:
        return self.Phase.CHASING

    def create_phase_machine(self, phase: Phase) -> StateMachine[RobotContext]:
        if phase == Midfielder.Phase.CHASING:
            return ChasingBallStateMachine(ROLE_AREA)
        elif phase == Midfielder.Phase.INTERCEPTING:
            return InterceptionStateMachine(ROLE_AREA)
        elif phase == Midfielder.Phase.PASSING:
            return PassingStateMachine(ROLE_AREA)
        elif phase == Midfielder.Phase.SHOOTING:
            return ShootingStateMachine(ROLE_AREA, force_center=self._force_center)
        elif phase == Midfielder.Phase.RECEIVING:
            return ReceivingStateMachine(ROLE_AREA)
        elif phase == Midfielder.Phase.SHADOW:
            return ShadowBallStateMachine(ROLE_AREA)
    
    def update_phase(self, robot, context: RobotContext):

        bx, by = context.ball.position[:2]
        ball_in_zone = ROLE_AREA.is_point_legal(context, bx, by)
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

        teammate_has_ball = any(
            check_possession(context.ball.position, t.position)
            for t in context.teammates.values()
        )

        opponent_has_ball = any(
            check_possession(context.ball.position, o.position)
            for o in context.opponents.values()
        )

        # --------------------------------------------------
        # 1) Midfielder has the ball → PASSING or SHOOTING
        # --------------------------------------------------
        if context.has_possession:
            # Fallback: if held for > 3s in any state, ensure we are in SHOOTING
            if self._possession_start is not None and (time.time() - self._possession_start > 3.0):
                self._force_center = True
                if robot.current_phase != self.Phase.SHOOTING:
                    robot.switch_phase(self.Phase.SHOOTING)
                return
            
            # Reset force_center if we just entered possession or are within normal time
            if self._possession_start is not None and (time.time() - self._possession_start <= 3.0):
                self._force_center = False

            if robot.current_phase not in [self.Phase.PASSING, self.Phase.SHOOTING]:
                best_pass = select_best_pass_target(context)
                if best_pass is not None:
                    context.preferred_teammate = best_pass
                    robot.switch_phase(self.Phase.PASSING)
                else:
                    # Fallback to shooting if no good pass targets
                    robot.switch_phase(self.Phase.SHOOTING)
            return

        # --------------------------------------------------
        # 2) Teammate has the ball → RECEIVING/INTERCEPTING
        # --------------------------------------------------
        if teammate_has_ball:
            # If we were in RECEIVING or chasing, switch to INTERCEPTING
            # to avoid fighting and cover opponents
            if robot.current_phase != self.Phase.INTERCEPTING:
                robot.switch_phase(self.Phase.INTERCEPTING)
            return

        # --------------------------------------------------
        # 3) Opponent has the ball OR fast ball → INTERCEPTING
        # --------------------------------------------------
        if opponent_has_ball or ball_speed > 500.0:
            if robot.current_phase != self.Phase.INTERCEPTING:
                robot.switch_phase(self.Phase.INTERCEPTING)
            return

        # --------------------------------------------------
        # 4) Ball inside zone & free → CHASING (only if closest)
        # --------------------------------------------------
        if ball_in_zone:
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
                    robot.switch_phase(self.Phase.CHASING)
                return
            else:
                if robot.current_phase != self.Phase.INTERCEPTING:
                    robot.switch_phase(self.Phase.INTERCEPTING)
                return

        # --------------------------------------------------
        # 5) Ball outside zone → SHADOW
        # --------------------------------------------------
        if robot.current_phase != self.Phase.SHADOW:
            robot.switch_phase(self.Phase.SHADOW)

def select_best_pass_target(context: RobotContext) -> Optional[int]:
    """
    Select the best teammate ID to pass to. 
    Prefer teammates at a reasonable passing distance.
    If everyone is too far (> 3500mm), return None to fallback to shooting.
    """
    my_pos = context.my_state.position[:2]
    best_tid = None
    max_dist = -1.0

    # Candidates: all teammates
    for tid, teammate in context.teammates.items():
        dist = np.linalg.norm(teammate.position[:2] - my_pos)
        
        # Don't pass if they are too close (within 800mm) or too far (> 3500mm)
        if dist < 500.0 :
            continue
            
        if dist > max_dist:
            max_dist = dist
            best_tid = tid

    return best_tid
