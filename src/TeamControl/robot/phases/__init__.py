"""Phases package - reusable state machine implementations."""

from TeamControl.robot.phases.ball_handling import (
    ChasingBallStateMachine,
    PassingStateMachine,
    ReceivingStateMachine,
    StrikerStateMachine,
    ShootingStateMachine,
)

from TeamControl.robot.phases.basic import (
    HoldPositionFaceBallStateMachine,
    InterceptionStateMachine,
    SupportBehindBallStateMachine,
    GoalieStateMachine,
)

__all__ = [
    # Ball handling
    "ChasingBallStateMachine",
    "PassingStateMachine",
    "ReceivingStateMachine",
    "StrikerStateMachine",
    "ShootingStateMachine",

    # Basic roles
    "HoldPositionFaceBallStateMachine",
    "InterceptionStateMachine",
    "SupportBehindBallStateMachine",
    "GoalieStateMachine",
]
