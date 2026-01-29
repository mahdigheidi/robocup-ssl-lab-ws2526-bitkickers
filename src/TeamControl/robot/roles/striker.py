#stiker role

from enum import Enum, auto
import time
import logging
from TeamControl.SSL.grSim.passing_process import check_possession
from TeamControl.robot import context
from TeamControl.robot.roles import goalie
import numpy as np

from TeamControl.robot.role import Role
from TeamControl.robot.context import RobotContext
from TeamControl.utils.state_machine import StateMachine

from TeamControl.robot.phases.basic import (
    InterceptionStateMachine,
)

from TeamControl.robot.phases import (
    ReceivingStateMachine,
    ChasingBallStateMachine,
    StrikerStateMachine,
)


BALL_SLOW_THRESHOLD = 500
BALL_DIRECTION_DOT_THRESHOLD = 0.3
GOALIE_HOLD_TIME = 1.0

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
        RECEIVING = auto()
        PLAY = auto()
        CHASING = auto()
        INTERCEPT = auto()

    def __init__(self, blue_goal= None):
        super().__init__()
        self._goalie_hold_start = None
        self.blue_goal = blue_goal

    def get_initial_phase(self):
        return self.Phase.RECEIVING

    
    def create_phase_machine(self, phase):

        if phase == self.Phase.RECEIVING:
            return ReceivingStateMachine()

        if phase == self.Phase.PLAY:
            return StrikerStateMachine()

        if phase == self.Phase.CHASING:
            return ChasingBallStateMachine()

        if phase == self.Phase.INTERCEPT:
            return InterceptionStateMachine()

        raise ValueError(phase)
    

    def ball_moving_toward_me(self, context: RobotContext) -> bool:
        ball_pos = context.ball.position
        ball_vel = context.ball.velocity
        my_pos = context.my_state.position

        speed = np.linalg.norm(ball_vel)
        if speed < BALL_SLOW_THRESHOLD:
            return False

        ball_dir = ball_vel / speed
        to_me = my_pos[:2] - ball_pos
        dist = np.linalg.norm(to_me)

        if dist < 1:
            return False

        return np.dot(ball_dir, to_me / dist) > BALL_DIRECTION_DOT_THRESHOLD

    def opp_goalie_has_ball(self, context: RobotContext) -> bool:
        if not context.opponents:
            return False

        goal_op = -4500.0 if context.is_yellow else 4500.0

        goalie = min(
            context.opponents.values(),
            key=lambda o: abs(o.position[0] - goal_op)
        )

        return check_possession(context.ball.position, goalie.position)

    def update_phase(self, robot, context: RobotContext):

        # EXIT intercept cleanly
        if robot.current_phase == self.Phase.INTERCEPT:
            if not self.opp_goalie_has_ball(context):
                log_phase(robot, self.Phase.RECEIVING, "opponent goalie released ball")
                return self.Phase.RECEIVING

        has_ball = check_possession(context.ball.position, context.my_state.position)
        ball_pos = context.ball.position
        teammate_has_ball = any(
            check_possession(ball_pos, t.position)
            for t in context.teammates.values()
        )

        # 1. Goalie hold → intercept
        if self.opp_goalie_has_ball(context):
            if self._goalie_hold_start is None:
                self._goalie_hold_start = time.time()
                #logger.debug(
                #f"[R{robot.robot_id}] goalie possession detected, starting timer"
            #)
            elif time.time() - self._goalie_hold_start > GOALIE_HOLD_TIME:
                log_phase(robot, self.Phase.INTERCEPT, "opponent goalie holding ball")
                return self.Phase.INTERCEPT
        else:
            self._goalie_hold_start = None

        # 2. have ball
        if has_ball:
            log_phase(robot, self.Phase.PLAY, "self has possession")
            return self.Phase.PLAY

        # 3. Teammate has ball
        if teammate_has_ball:
            #log_phase(robot, self.Phase.RECEIVING, "teammate has possession")
            return self.Phase.RECEIVING

        # 5. Otherwise chase
        #log_phase(robot, self.Phase.CHASING, "no possession, chasing ball")
        return self.Phase.CHASING
