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
from TeamControl.robot.context import RobotContext, teammate_can_chase_ball
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
    x_min=-4500.0,
    x_max=-1500.0,
    y_min=-2900.0,
    y_max=2900.0,
    defense_x=-3450.0, # goal box edge
    defense_y_half=1050.0,
)

ROLE_AREA = STRIKER_AREA

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
        self._force_center = False

    def get_initial_phase(self):
        return self.Phase.CHASING

    
    def create_phase_machine(self, phase):

        if phase == self.Phase.RECEIVING:
            return ReceivingStateMachine(ROLE_AREA)

        if phase == self.Phase.PLAY:
            return StrikerStateMachine(ROLE_AREA)

        if phase == self.Phase.CHASING:
            return ChasingBallStateMachine(ROLE_AREA)

       # if phase == self.Phase.HOLD:
           # return HoldPositionFaceBallStateMachine(self.home)

        if phase == self.Phase.INTERCEPT:
            return InterceptionStateMachine(STRIKER_AREA)
        
        if phase == self.Phase.KICK_TO_GOAL:
            return KickToGoalStateMachine(ROLE_AREA, force_center=self._force_center)
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



# Phase Update Logic

    def update_phase(self, robot, context: RobotContext):
        
        bx, by = context.ball.position[:2]
        ball_legal = STRIKER_AREA.is_point_legal(context, bx, by)
        
        teammate_has_ball = any(
            check_possession(context.ball.position, t.position)
            for t in context.teammates.values()
        )

        opponent_has_ball = any(
            check_possession(context.ball.position, o.position)
            for o in context.opponents.values()
        )
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
                        if robot.current_phase != self.Phase.INTERCEPT:
                            robot.switch_phase(self.Phase.INTERCEPT)
                        return
        else:
            if not context.has_possession:
                self._possession_start = None

        if teammate_has_ball:
            if robot.current_phase != self.Phase.RECEIVING:
                robot.switch_phase(self.Phase.RECEIVING)
            return
        
        if context.has_possession:
            # Fallback: if held for > 3s in any state, ensure we are in KICK_TO_GOAL
            if self._possession_start is not None and (time.time() - self._possession_start > 3.0):
                self._force_center = True
                if robot.current_phase != self.Phase.KICK_TO_GOAL:
                    robot.switch_phase(self.Phase.KICK_TO_GOAL)
                return
            
            # Reset force_center if we just entered possession or are within normal time
            if self._possession_start is not None and (time.time() - self._possession_start <= 3.0):
                self._force_center = False

            if robot.current_phase != self.Phase.KICK_TO_GOAL:
                robot.switch_phase(self.Phase.KICK_TO_GOAL)
            return

        if ball_speed > 500.0 or self.opp_goalie_has_ball(context) or opponent_has_ball:
            if robot.current_phase != self.Phase.INTERCEPT:
                robot.switch_phase(self.Phase.INTERCEPT)
            return

        if ball_legal:
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
            
            
        if robot.current_phase != self.Phase.SHADOW:
            robot.switch_phase(self.Phase.SHADOW)
   