import logging
import random
import time
import numpy as np
from TeamControl.ball.ball_idle_spawner import BallIdleSpawner
from TeamControl.network.ssl_sockets import grSimSender
from TeamControl.ball.send_ball_grsim import send_ball_to_grsim
from TeamControl.world.model import WorldModel
from TeamControl.robot.robot import Robot
from TeamControl.robot.roles import PassingDemo
from TeamControl.robot.context import check_possession


# ANSI color codes
class Colors:
    # Bold colors (for special messages like pass completion)
    BOLD_GREEN = '\033[1;92m'
    BOLD_CYAN = '\033[1;96m'
    BOLD_YELLOW = '\033[1;93m'
    BOLD_MAGENTA = '\033[1;95m'
    BOLD_BLUE = '\033[1;94m'
    
    # Regular colors (for logging)
    CYAN = '\033[96m'           # Current: PHASE transitions
    YELLOW = '\033[93m'         # Current: state transitions
    GREEN = '\033[92m'          # Alternative option
    MAGENTA = '\033[95m'        # Alternative option
    BLUE = '\033[94m'           # Alternative option
    RED = '\033[91m'            # Alternative option
    
    RESET = '\033[0m'


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    
    def format(self, record):
        # Apply regular (non-bold) colors based on content
        if 'PHASE:' in record.getMessage():
            # Phase transitions in magenta
            record.msg = f"{Colors.MAGENTA}{record.msg}{Colors.RESET}"
        elif 'state:' in record.getMessage():
            # State transitions in cyan
            record.msg = f"{Colors.CYAN}{record.msg}{Colors.RESET}"
        
        return super().format(record)


# Configure logging for robot modules ONLY - don't touch root logger
robot_logger = logging.getLogger('TeamControl.robot')
robot_logger.setLevel(logging.INFO)
robot_handler = logging.StreamHandler()
robot_handler.setFormatter(ColoredFormatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))
robot_logger.addHandler(robot_handler)
robot_logger.propagate = False  # Don't propagate to root logger

state_logger = logging.getLogger('TeamControl.utils.state_machine')
state_logger.setLevel(logging.DEBUG)
state_logger.addHandler(robot_handler)
state_logger.propagate = False  # Don't propagate to root logger

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
    SIM_IP = "127.0.0.1"
    CMD_LISTEN_PORT = 20011
    TIME_LIMIT = 120

    sender = grSimSender(ip=SIM_IP, port=CMD_LISTEN_PORT, is_yellow=True)
    
    # Create two autonomous robots with PassingDemo role
    robot0 = Robot(robot_id=0, is_yellow=True, role=PassingDemo())
    robot1 = Robot(robot_id=1, is_yellow=True, role=PassingDemo())

    version = 0

    ball_pos = np.array([0.0, 0.0], dtype=np.float32)
    prev_ball_pos = np.array([0.0, 0.0], dtype=np.float32)
    ball_velocity = np.array([0.0, 0.0], dtype=np.float32)
    blue_robot_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    last_update_time = time.time()

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

    while time.time() - start_time < TIME_LIMIT:
        if version < wm.get_version():
            version = wm.get_version()
            current_time = time.time()
            dt = current_time - last_update_time
            last_update_time = current_time

            try:
                robot0_pos_data: np.ndarray = wm.get_yellow_robots(isYellow=True, robot_id=0).position  # type: ignore
                robot1_pos_data: np.ndarray = wm.get_yellow_robots(isYellow=True, robot_id=1).position  # type: ignore
                blue_robot_pos: np.ndarray = wm.get_yellow_robots(
                    isYellow=False, robot_id=0
                ).position  # type: ignore

                # Update robot positions
                robot0.update_position(robot0_pos_data)
                robot1.update_position(robot1_pos_data)

                prev_ball_pos = ball_pos.copy()
                ball_pos = wm.get_latest_frame().ball.position

                # Calculate ball velocity (mm/s)
                if dt > 0:
                    ball_velocity = (ball_pos - prev_ball_pos) / dt
                else:
                    ball_velocity = np.array([0.0, 0.0], dtype=np.float32)
            except Exception:
                pass

        # ---- DETERMINE BALL RESPONSIBILITY AND SWITCH PHASES ----
        passer_id = determine_ball_responsibility(
            robot0.position, robot1.position, ball_pos, ball_velocity
        )

        # Switch phases based on ball responsibility
        if passer_id == 0:
            # Robot 0 is responsible - check if it should chase or pass
            if check_possession(ball_pos, robot0.position):
                robot0.switch_phase(PassingDemo.Phase.PASSING)
            else:
                robot0.switch_phase(PassingDemo.Phase.CHASING)
            robot1.switch_phase(PassingDemo.Phase.RECEIVING)
        else:
            # Robot 1 is responsible - check if it should chase or pass
            if check_possession(ball_pos, robot1.position):
                robot1.switch_phase(PassingDemo.Phase.PASSING)
            else:
                robot1.switch_phase(PassingDemo.Phase.CHASING)
            robot0.switch_phase(PassingDemo.Phase.RECEIVING)

        # ---- EXECUTE ROBOT BEHAVIORS ----
        robot0.execute_behavior(
            sender=sender,
            spawner=spawner,
            passing_min_distance=PASSING_MIN_DISTANCE,
            other_pos=robot1.position,
            ball_pos=ball_pos,
            ball_velocity=ball_velocity,
        )
        robot1.execute_behavior(
            sender=sender,
            spawner=spawner,
            passing_min_distance=PASSING_MIN_DISTANCE,
            other_pos=robot0.position,
            ball_pos=ball_pos,
            ball_velocity=ball_velocity,
        )

        # ---- PASS COUNTING ----
        has_robot0 = check_possession(ball_pos, robot0.position)
        has_robot1 = check_possession(ball_pos, robot1.position)
        has_blue = distance(ball_pos, blue_robot_pos) < 120

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
