import random
import time
import numpy as np
from TeamControl.ball.ball_idle_spawner import BallIdleSpawner
from TeamControl.network.ssl_sockets import grSimSender
from TeamControl.ball.send_ball_grsim import send_ball_to_grsim
from TeamControl.world.model import WorldModel
from TeamControl.robot.robot import Robot
from TeamControl.robot.roles import Passer
from TeamControl.robot.context import WorldState, POSSESSION_THRESHOLD
from TeamControl.robot import enable_logging, Colors


def check_possession(ball_pos: np.ndarray, robot_pos: np.ndarray, threshold: float = POSSESSION_THRESHOLD) -> bool:
    """Check if robot has possession of the ball based on distance threshold."""
    dist = float(np.linalg.norm(ball_pos - robot_pos[:2]))
    return dist < threshold


# --------------------------
# Ball idle spawner parameters
# --------------------------
DIVISION = "B"  # 'A' or 'B'

# Ball responsibility constants
BALL_SLOW_THRESHOLD = 500  # mm/s - below this, ball is considered stationary
BALL_DIRECTION_DOT_THRESHOLD = 0.3  # Cosine threshold for "moving towards" check


# ============================================================================
# Helper Functions
# ============================================================================


def distance(pos0: np.ndarray, pos1: np.ndarray) -> float:
    """Calculate Euclidean distance between two 2D positions."""
    return float(np.linalg.norm(pos0[:2] - pos1[:2]))


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


def passing_process(wm: WorldModel):
    # Configure robot logging
    enable_logging()
    
    SIM_IP = "127.0.0.1"
    CMD_LISTEN_PORT = 20011
    TIME_LIMIT = 120

    sender = grSimSender(ip=SIM_IP, port=CMD_LISTEN_PORT, is_yellow=True)
    
    # Create two autonomous robots with PassingDemo role
    robot0 = Robot(robot_id=0, is_yellow=True, role=Passer())
    robot1 = Robot(robot_id=1, is_yellow=True, role=Passer())

    # Create world state tracker
    world_state = WorldState()

    version = 0
    start_time = time.time()
    pass_count = 0
    last_possessor = None
    blue_possession_start = None
    BLUE_POSSESSION_DURATION = 0.5

    PASSING_MIN_DISTANCE = 3000

    spawner = BallIdleSpawner(division=DIVISION)

    # Spawn robots
    robot0.spawn(x=1500, y=-1500, orientation=0)
    time.sleep(0.1)
    robot1.spawn(x=1500, y=1500, orientation=0)
    time.sleep(0.1)
    
    # Spawn ball
    ball_x = random.uniform(1400, 1500)
    ball_y = random.uniform(-100, 100)
    send_ball_to_grsim(ball_x / 1000, ball_y / 1000, 0, 0)
    time.sleep(0.1)

    blue_robot_state = None  # Initialize outside loop

    while time.time() - start_time < TIME_LIMIT:
        if version < wm.get_version():
            version = wm.get_version()
            current_time = time.time()

            # Update world state from world model
            world_state.update(wm, current_time)

            try:
                # Get robot positions from world state
                robot0_state = world_state.get_robot_state(0, is_yellow=True)
                robot1_state = world_state.get_robot_state(1, is_yellow=True)
                blue_robot_state = world_state.get_robot_state(0, is_yellow=False)

                if robot0_state:
                    robot0.update_position(robot0_state.position)
                if robot1_state:
                    robot1.update_position(robot1_state.position)
            except Exception:
                pass

        # ---- DETERMINE BALL RESPONSIBILITY AND SWITCH PHASES ----
        # Get current ball and robot positions
        ball_pos = world_state.ball.position
        ball_velocity = world_state.ball.velocity
        
        passer_id = determine_ball_responsibility(
            robot0.position, robot1.position, ball_pos, ball_velocity
        )

        # Switch phases based on ball responsibility
        if passer_id == 0:
            # Robot 0 is responsible - check if it should chase or pass
            if check_possession(ball_pos, robot0.position):
                robot0.switch_phase(Passer.Phase.PASSING)
            else:
                robot0.switch_phase(Passer.Phase.CHASING)
            robot1.switch_phase(Passer.Phase.RECEIVING)
        else:
            # Robot 1 is responsible - check if it should chase or pass
            if check_possession(ball_pos, robot1.position):
                robot1.switch_phase(Passer.Phase.PASSING)
            else:
                robot1.switch_phase(Passer.Phase.CHASING)
            robot0.switch_phase(Passer.Phase.RECEIVING)

        # ---- EXECUTE ROBOT BEHAVIORS ----
        robot0.execute_behavior(
            world_state=world_state,
            sender=sender,
            spawner=spawner,
            passing_min_distance=PASSING_MIN_DISTANCE,
        )
        robot1.execute_behavior(
            world_state=world_state,
            sender=sender,
            spawner=spawner,
            passing_min_distance=PASSING_MIN_DISTANCE,
        )

        # ---- PASS COUNTING ----
        has_robot0 = check_possession(ball_pos, robot0.position)
        has_robot1 = check_possession(ball_pos, robot1.position)
        
        has_blue = False
        if blue_robot_state:
            has_blue = check_possession(ball_pos, blue_robot_state.position)

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
            print(f"{Colors.BOLD_GREEN}Pass completed! Total passes: {pass_count}{Colors.RESET}")

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
