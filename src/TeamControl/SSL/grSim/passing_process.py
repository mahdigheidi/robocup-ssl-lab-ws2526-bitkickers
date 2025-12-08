import random
import time
from typing import Optional, Any, Type
from dataclasses import dataclass
import numpy as np
from TeamControl.ball.ball_idle_spawner import BallIdleSpawner
from TeamControl.network.ssl_sockets import grSimSender
from TeamControl.network.robot_command import RobotCommand
from TeamControl.robot.Movement import RobotMovement
from TeamControl.robot.spawn_robot import send_robot_to_grsim
from TeamControl.ball.send_ball_grsim import send_ball_to_grsim
from TeamControl.world.model import WorldModel
from TeamControl.utils.state_machine import State, StateMachine

# --------------------------
# Ball idle spawner parameters
# --------------------------
DIVISION = "B"  # 'A' or 'B'

# Ball responsibility constants
BALL_SLOW_THRESHOLD = 500  # mm/s - below this, ball is considered stationary
BALL_DIRECTION_DOT_THRESHOLD = 0.3  # Cosine threshold for "moving towards" check


@dataclass
class PassingContext:
    """Context object containing all information needed for state transitions."""

    passer_id: int
    receiver_id: int
    passer_pos: np.ndarray
    passer_movement: RobotMovement
    receiver_pos: np.ndarray
    receiver_movement: RobotMovement
    ball_pos: np.ndarray
    ball_velocity: np.ndarray
    has_possession: bool

    @classmethod
    def from_robots(
        cls,
        robot0_pos: np.ndarray,
        robot0_movement: RobotMovement,
        robot1_pos: np.ndarray,
        robot1_movement: RobotMovement,
        ball_pos: np.ndarray,
        ball_velocity: np.ndarray,
        locked_passer_id: Optional[int] = None,
    ) -> "PassingContext":
        """Create context from robot states."""
        # Determine passer and receiver
        if locked_passer_id is not None:
            passer_id = locked_passer_id
        else:
            passer_id = determine_ball_responsibility(
                robot0_pos, robot1_pos, ball_pos, ball_velocity
            )
        receiver_id = 1 - passer_id

        # Set passer and receiver data
        passer_pos = robot0_pos if passer_id == 0 else robot1_pos
        passer_movement = robot0_movement if passer_id == 0 else robot1_movement
        receiver_pos = robot1_pos if passer_id == 0 else robot0_pos
        receiver_movement = robot1_movement if passer_id == 0 else robot0_movement

        # Check possession
        has_possession = check_possession(ball_pos, passer_pos)

        return cls(
            passer_id=passer_id,
            receiver_id=receiver_id,
            passer_pos=passer_pos,
            passer_movement=passer_movement,
            receiver_pos=receiver_pos,
            receiver_movement=receiver_movement,
            ball_pos=ball_pos,
            ball_velocity=ball_velocity,
            has_possession=has_possession,
        )


# ============================================================================
# State Classes
# ============================================================================


class ChasingBallState(State[PassingContext]):
    """State where robot chases the ball from far away - fast approach."""

    def handle_transition(self, context: PassingContext) -> Optional[Type[State[PassingContext]]]:
        ball_dist = distance(context.ball_pos, context.passer_pos)

        # Close enough to switch to gentle approach
        if ball_dist < 200:
            return ApproachingBallState

        return None

    def execute(self, context: PassingContext, **kwargs: Any) -> None:
        sender: grSimSender = kwargs["sender"]
        spawner: BallIdleSpawner = kwargs["spawner"]
        passing_min_distance: float = kwargs["passing_min_distance"]

        # PASSER: Chase the ball quickly
        vx, vy = context.passer_movement.go_to(
            context.passer_pos,
            context.ball_pos,
            kicker_to_target=True,
            max_acceleration=1300,
            max_speed=5000,  # Fast approach
        )
        w = context.passer_movement.face_towards(
            robot_pos=context.passer_pos, target=context.ball_pos
        )
        sender.send_command(
            RobotCommand(robot_id=context.passer_id, vx=vx, vy=vy, w=w, kick=0, dribble=1)
        )

        # RECEIVER: Position at minimum distance
        target_pos = calculate_receiver_position(
            context.passer_pos,
            context.receiver_pos,
            passing_min_distance,
            spawner.field_x_min,
            spawner.field_x_max,
            spawner.field_y_min,
            spawner.field_y_max,
        )
        vx, vy = context.receiver_movement.go_to(context.receiver_pos, target_pos)
        w = context.receiver_movement.face_towards(context.receiver_pos, context.passer_pos)
        sender.send_command(
            RobotCommand(robot_id=context.receiver_id, vx=vx, vy=vy, w=w, kick=0, dribble=0)
        )


class ApproachingBallState(State[PassingContext]):
    """State where robot approaches ball gently when close - prevents bumping."""

    def handle_transition(self, context: PassingContext) -> Optional[Type[State[PassingContext]]]:
        ball_dist = distance(context.ball_pos, context.passer_pos)

        # Ball moved far away - go back to fast chasing
        if ball_dist > 250:
            return ChasingBallState

        # Got possession - start aligning
        if context.has_possession:
            return AligningWithTargetState

        return None

    def execute(self, context: PassingContext, **kwargs: Any) -> None:
        sender: grSimSender = kwargs["sender"]
        spawner: BallIdleSpawner = kwargs["spawner"]
        passing_min_distance: float = kwargs["passing_min_distance"]

        # PASSER: Approach ball gently to avoid bumping
        vx, vy = context.passer_movement.go_to(
            context.passer_pos,
            context.ball_pos,
            kicker_to_target=True,
            max_acceleration=800,
            max_speed=1500,  # Gentle approach
        )
        w = context.passer_movement.face_towards(
            robot_pos=context.passer_pos, target=context.ball_pos
        )
        sender.send_command(
            RobotCommand(robot_id=context.passer_id, vx=vx, vy=vy, w=w, kick=0, dribble=1)
        )

        # RECEIVER: Position at minimum distance
        target_pos = calculate_receiver_position(
            context.passer_pos,
            context.receiver_pos,
            passing_min_distance,
            spawner.field_x_min,
            spawner.field_x_max,
            spawner.field_y_min,
            spawner.field_y_max,
        )
        vx, vy = context.receiver_movement.go_to(context.receiver_pos, target_pos)
        w = context.receiver_movement.face_towards(context.receiver_pos, context.passer_pos)
        sender.send_command(
            RobotCommand(robot_id=context.receiver_id, vx=vx, vy=vy, w=w, kick=0, dribble=0)
        )


class AligningWithTargetState(State[PassingContext]):
    """State where robot aligns with target to pass."""

    def handle_transition(self, context: PassingContext) -> Optional[Type[State[PassingContext]]]:
        # Lost possession - go back to approaching
        if not context.has_possession:
            return ApproachingBallState

        w = context.passer_movement.face_towards(
            robot_pos=context.passer_pos,
            target=context.receiver_pos,
            max_angular_acceleration=3,
        )
        if w == 0:
            return KickingState
        return None

    def execute(self, context: PassingContext, **kwargs: Any) -> None:
        sender: grSimSender = kwargs["sender"]

        # PASSER: Align with target robot, dribbler on
        vx, vy = 0, 0
        w = context.passer_movement.face_towards(
            robot_pos=context.passer_pos,
            target=context.receiver_pos,
            max_angular_acceleration=3,
        )
        sender.send_command(
            RobotCommand(robot_id=context.passer_id, vx=vx, vy=vy, w=w, kick=0, dribble=1)
        )


class KickingState(State[PassingContext]):
    """State where robot kicks the ball."""

    def handle_transition(self, context: PassingContext) -> Optional[Type[State[PassingContext]]]:
        if not context.has_possession:
            return ChasingBallState
        return None

    def execute(self, context: PassingContext, **kwargs: Any) -> None:
        sender: grSimSender = kwargs["sender"]

        # PASSER: Kick the ball
        vx, vy = 0, 0
        w = 0
        sender.send_command(
            RobotCommand(robot_id=context.passer_id, vx=vx, vy=vy, w=w, kick=5, dribble=0)
        )


# ============================================================================
# State Machine Wrapper
# ============================================================================


class PassingStateMachine:
    """Wrapper for passing state machine with role switching logic."""

    def __init__(self):
        # Create states
        chasing = ChasingBallState()
        approaching = ApproachingBallState()
        aligning = AligningWithTargetState()
        kicking = KickingState()

        # Create state machine
        self._machine = StateMachine[PassingContext](chasing)
        self._machine.register_state(approaching)
        self._machine.register_state(aligning)
        self._machine.register_state(kicking)

        self.current_passer_id: Optional[int] = None

    def _handle_role_switch(self, context: PassingContext) -> None:
        """Reset state machine when passer role switches."""
        if self.current_passer_id is not None and self.current_passer_id != context.passer_id:
            from TeamControl.utils.state_machine import _START_TIME

            elapsed = time.time() - _START_TIME
            print(
                f"[STATE @ {elapsed:.3f}s] Role switched: Robot {self.current_passer_id} -> "
                f"Robot {context.passer_id}"
            )
            self._machine.force_transition(ChasingBallState, context)
        self.current_passer_id = context.passer_id

    def update(
        self,
        robot0_pos: np.ndarray,
        robot0_movement: RobotMovement,
        robot1_pos: np.ndarray,
        robot1_movement: RobotMovement,
        ball_pos: np.ndarray,
        ball_velocity: np.ndarray,
    ) -> tuple[Type[State[PassingContext]], int, int]:
        """
        Update state machine based on current conditions.

        Returns: (current_state_type, passer_id, receiver_id)
        """
        # Build context
        context = PassingContext.from_robots(
            robot0_pos,
            robot0_movement,
            robot1_pos,
            robot1_movement,
            ball_pos,
            ball_velocity,
        )

        # Handle role switching
        self._handle_role_switch(context)

        # Update state machine
        self._machine.update(context)

        return self._machine.get_current_state_type(), context.passer_id, context.receiver_id

    def execute_state(
        self,
        context: PassingContext,
        sender: grSimSender,
        spawner: BallIdleSpawner,
        passing_min_distance: float,
    ):
        """Execute behavior for the current state."""
        self._machine.execute(
            context,
            sender=sender,
            spawner=spawner,
            passing_min_distance=passing_min_distance,
        )


# ============================================================================
# Helper Functions
# ============================================================================


def distance(pos0: np.ndarray, pos1: np.ndarray) -> float:
    """Calculate Euclidean distance between two 2D positions."""
    return float(np.linalg.norm(pos0[:2] - pos1[:2]))


def check_possession(ball_pos: np.ndarray, robot_pos: np.ndarray, threshold: float = 120) -> bool:
    """Check if robot has possession of the ball based on distance threshold."""
    d = distance(ball_pos, robot_pos)
    return d != 0 and d < threshold


def determine_ball_responsibility(
    robot0_pos: np.ndarray,
    robot1_pos: np.ndarray,
    ball_pos: np.ndarray,
    ball_velocity: np.ndarray,
) -> int:
    """
    Determine which robot should be responsible for the ball.

    If ball is slow/stationary: closer robot is responsible
    If ball is moving fast: robot toward whom ball is moving is responsible

    Returns: 0 or 1 (robot ID)
    """
    ball_speed = np.linalg.norm(ball_velocity)

    if ball_speed < BALL_SLOW_THRESHOLD:
        # Ball is stationary or slow - use distance
        dist0 = distance(robot0_pos, ball_pos)
        dist1 = distance(robot1_pos, ball_pos)
        return 0 if dist0 < dist1 else 1
    else:
        # Ball is moving - check which robot it's moving towards
        ball_direction = ball_velocity / ball_speed

        # Vectors from ball to each robot
        to_robot0 = robot0_pos[:2] - ball_pos
        to_robot1 = robot1_pos[:2] - ball_pos

        # Normalize
        dist0 = np.linalg.norm(to_robot0)
        dist1 = np.linalg.norm(to_robot1)

        if dist0 > 0:
            to_robot0 = to_robot0 / dist0
        if dist1 > 0:
            to_robot1 = to_robot1 / dist1

        # Dot product: positive means moving towards
        dot0 = np.dot(ball_direction, to_robot0)
        dot1 = np.dot(ball_direction, to_robot1)

        # If clearly moving towards one robot, that robot is responsible
        if dot0 > BALL_DIRECTION_DOT_THRESHOLD and dot0 > dot1:
            return 0
        elif dot1 > BALL_DIRECTION_DOT_THRESHOLD and dot1 > dot0:
            return 1
        else:
            # Not clearly moving towards either - use distance
            return 0 if dist0 < dist1 else 1


def calculate_receiver_position(
    passer_pos: np.ndarray,
    receiver_pos: np.ndarray,
    min_distance: float,
    field_x_min: float,
    field_x_max: float,
    field_y_min: float,
    field_y_max: float,
    margin: float = 200,
) -> np.ndarray:
    """
    Calculate position for receiver robot at minimum distance from passer.
    Ensures receiver stays within field boundaries.
    """
    delta = receiver_pos[:2] - passer_pos[:2]
    dist = np.linalg.norm(delta)

    if dist == 0 or dist >= min_distance:
        return receiver_pos[:2]

    # Calculate target position at min_distance
    target = passer_pos[:2] + (delta / dist) * min_distance

    # Clamp to field boundaries
    clamped = np.array(
        [
            max(field_x_min + margin, min(field_x_max - margin, target[0])),
            max(field_y_min + margin, min(field_y_max - margin, target[1])),
        ]
    )

    # If clamping brought us closer than min_distance, project back onto circle
    clamped_delta = clamped - passer_pos[:2]
    clamped_dist = np.linalg.norm(clamped_delta)

    if clamped_dist < min_distance:
        if clamped_dist > 0:
            target = passer_pos[:2] + (clamped_delta / clamped_dist) * min_distance
        else:
            target = clamped
    else:
        target = clamped

    return target


def passing_process(wm: WorldModel):
    SIM_IP = "127.0.0.1"
    CMD_LISTEN_PORT = 20011
    TIME_LIMIT = 120

    sender = grSimSender(ip=SIM_IP, port=CMD_LISTEN_PORT, is_yellow=True)
    robot0_movement = RobotMovement()
    robot1_movement = RobotMovement()

    version = 0

    robot0_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    robot1_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    blue_robot_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    ball_pos = np.array([0.0, 0.0], dtype=np.float32)
    prev_ball_pos = np.array([0.0, 0.0], dtype=np.float32)
    ball_velocity = np.array([0.0, 0.0], dtype=np.float32)
    last_update_time = time.time()

    start_time = time.time()
    pass_count = 0
    last_possessor = None
    blue_possession_start = None
    BLUE_POSSESSION_DURATION = 0.5

    PASSING_MIN_DISTANCE = 3000

    spawner = BallIdleSpawner(division=DIVISION)
    state_machine = PassingStateMachine()

    send_robot_to_grsim(team_yellow=True, robot_id=0, x=1500, y=-1500, orientation=0)
    time.sleep(0.1)
    send_robot_to_grsim(team_yellow=True, robot_id=1, x=1500, y=1500, orientation=0)
    time.sleep(0.1)
    ball_x = random.uniform(1400, 1500)
    ball_y = random.uniform(-100, 100)
    send_ball_to_grsim(ball_x / 1000, ball_y / 1000, 0, 0)
    time.sleep(0.1)

    while time.time() - start_time < TIME_LIMIT:
        if version < wm.get_version():
            version = wm.get_version()
            current_time = time.time()
            dt = current_time - last_update_time
            last_update_time = current_time

            try:
                robot0_pos: np.ndarray = wm.get_yellow_robots(isYellow=True, robot_id=0).position  # type: ignore
                robot1_pos: np.ndarray = wm.get_yellow_robots(isYellow=True, robot_id=1).position  # type: ignore
                blue_robot_pos: np.ndarray = wm.get_yellow_robots(
                    isYellow=False, robot_id=0
                ).position  # type: ignore

                prev_ball_pos = ball_pos.copy()
                ball_pos = wm.get_latest_frame().ball.position

                # Calculate ball velocity (mm/s)
                if dt > 0:
                    ball_velocity = (ball_pos - prev_ball_pos) / dt
                else:
                    ball_velocity = np.array([0.0, 0.0], dtype=np.float32)
            except Exception:
                pass

        # ---- PASSING ----

        # Update and execute state machine
        _, passer_id, receiver_id = state_machine.update(
            robot0_pos=robot0_pos,
            robot0_movement=robot0_movement,
            robot1_pos=robot1_pos,
            robot1_movement=robot1_movement,
            ball_pos=ball_pos,
            ball_velocity=ball_velocity,
        )

        # Build context for execution
        context = PassingContext.from_robots(
            robot0_pos, robot0_movement, robot1_pos, robot1_movement, ball_pos, ball_velocity
        )

        state_machine.execute_state(
            context=context,
            sender=sender,
            spawner=spawner,
            passing_min_distance=PASSING_MIN_DISTANCE,
        )

        # ---- PASS COUNTING ----

        has_robot0 = check_possession(ball_pos, robot0_pos)
        has_robot1 = check_possession(ball_pos, robot1_pos)
        has_blue = check_possession(ball_pos, blue_robot_pos)

        current_possessor = None
        if has_robot0:
            current_possessor = 0
        elif has_robot1:
            current_possessor = 1

        if (
            last_possessor is not None
            and current_possessor is not None
            and last_possessor != current_possessor
        ):
            pass_count += 1
            print(f"Pass completed! Total passes: {pass_count}")

        if current_possessor is not None:
            last_possessor = current_possessor

        # ---- STOPPING CONDITIONS ----
        if has_blue:
            if blue_possession_start is None:
                blue_possession_start = time.time()
            elif time.time() - blue_possession_start >= BLUE_POSSESSION_DURATION:
                print("Blue robot intercepted the ball!")
                break
        else:
            blue_possession_start = None

        if not spawner.is_ball_in_field(ball_pos):
            print("Ball is out of field!")
            break

    print(f"Final result: {pass_count} passes in {time.time() - start_time:.1f} seconds")
