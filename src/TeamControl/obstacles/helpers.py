import math
from shapely.geometry import Point, Polygon
from TeamControl.robot.context import RobotContext, WorldState, POSSESSION_THRESHOLD
import numpy as np

from dataclasses import dataclass
# ----------------------------------------------------
# ------------ Constants and Polygons ----------------
# ----------------------------------------------------
ROBOT_RADIUS = 88.5 # m
COLLISION_DISTANCE = ROBOT_RADIUS * 2 

DEFENSE_AREA_POLYGON = [
    (3500, 1000),  # top-left
    (4500, 1000),  # top-right
    (4500, -1000), # bottom-right
    (3500, -1000), # bottom-left
]

FIELD_POLYGON = Polygon([
    (-4500, -3000),  # bottom-left corner
    (-4500, 3000),   # top-left corner
    (4500, 3000),    # top-right corner
    (4500, -3000),   # bottom-right corner
])

# ----------------------------------------------------
# --------------- GEOMETRY HELPERS -------------------
# ----------------------------------------------------
def distance(p1, p2):
    """Euclidean distance between points p1 and p2."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# ----------------------------------------------------
# -------- STOP CONDITION CHECKING HELPERS -----------
# ----------------------------------------------------
def point_in_defense_area(agent_pos, robot_radius=ROBOT_RADIUS, allow=False):
    """
    Check if the robot body (circle with radius) intersects the opponent defense area polygon.

    Args:
        agent_pos (tuple): (x, y) robot center position.
        robot_radius (float): radius of the robot.

    Returns:
        bool: True if any part of the robot is inside the defense area.
    """
    if allow:
        return False
    robot_circle = Point(agent_pos).buffer(robot_radius)
    defense_area_poly = Polygon(DEFENSE_AREA_POLYGON)
    return robot_circle.intersects(defense_area_poly)

def check_collisions(agent_pos, other_robot_pos):
    """
    Returns True if agent robot is touching another robot.
    Touching means distance between centers <= 2*robot radius.
    """
    for robot_position in other_robot_pos:
        if distance(agent_pos, robot_position) <= COLLISION_DISTANCE:
            return True
    return False

def has_reached_target(robot_pos, target_pos, threshold=5):
    """
    Checks if the robot is within `threshold` distance of the target position.

    Args:
        robot_pos (tuple): (x, y) current position of the robot.
        target_pos (tuple): (x, y) target position.
        threshold (float): Distance tolerance to consider as "reached". (in cm)

    Returns:
        bool: True if robot is within threshold distance of target, else False.
    """
    # TODO: Change threshold if needed
    dx = robot_pos[0] - target_pos[0]
    dy = robot_pos[1] - target_pos[1]
    dist = math.sqrt(dx * dx + dy * dy)
    return dist <= threshold

def is_in_field(position):
    """
    Check if a given (x, y) position is inside the soccer field.

    Args:
        position (tuple): (x, y) coordinates.

    Returns:
        bool: True if inside the field polygon, False otherwise.
    """
    point = Point(position)
    return FIELD_POLYGON.contains(point)

def check_stop_conditions(agent_pos, target_pos, static_robots, dynamic_robots, elapsed_time=0, time_limit=30, is_goalie=False):
    """
    Check all stop conditions and return (should_stop, message).

    Args:
        agent_pos (tuple): (x, y) position of the robot.
        target_pos (tuple): (x, y) target position.
        static_robots (list of tuples): List of (x, y) positions.
        dynamic_robots (list of tuples): List of (x, y) positions.
        elapsed_time (float): Seconds elapsed since scenario start.

    Returns:
        (bool, str): Tuple where first element is True if scenario should stop,
                     second element is a human-readable message.
    """

    if has_reached_target(agent_pos, target_pos):
        return True, f"Target reached in {elapsed_time:.2f} seconds.\n{time_limit-elapsed_time:.2f} POINTS, {agent_pos}"

    if not is_in_field(agent_pos):
        return True, "Robot left the field!\n0 POINTS"

    all_obstacles = static_robots + dynamic_robots
    if len(all_obstacles):
        if check_collisions(agent_pos, all_obstacles):
            return True, "Collision detected with another robot!\n0 POINTS"

    if point_in_defense_area(agent_pos, alllow=is_goalie):
        return True, "Robot entered opponent's defense area!\n0 POINTS"

    return False, ""

def ball_in_defense_area(ball_pos):
    return Point(ball_pos).within(Polygon(DEFENSE_AREA_POLYGON))


def check_possession(
    ball_pos: np.ndarray, robot_pos: np.ndarray, threshold: float = POSSESSION_THRESHOLD
) -> bool:
    """Check if robot has possession of the ball based on distance threshold."""
    dist = float(np.linalg.norm(ball_pos - robot_pos[:2]))
    return dist < threshold


# ----------------------------------------------------
# ---------------- ROLE AREA SYSTEM ------------------
# ----------------------------------------------------

@dataclass(frozen=True)
class RoleArea:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    defense_x: float | None = None
    defense_y_half: float | None = None

    def is_point_legal(self, context, x: float, y: float) -> bool:
        if not (self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max):
            return False

        if self.defense_x is not None and self.defense_y_half is not None:
            if context.is_yellow:
                in_defense = x >= self.defense_x and abs(y) <= self.defense_y_half
            else:
                in_defense = x <= self.defense_x and abs(y) <= self.defense_y_half

            if in_defense:
                return False

        return True


    def project_to_legal_area(
        self,
        context,
        x: float,
        y: float,
    ) -> np.ndarray:
        """
        Projects (x, y) into the nearest legal point inside the role area.
        """

        # 1) Clamp to rectangle
        cx = np.clip(x, self.x_min, self.x_max)
        cy = np.clip(y, self.y_min, self.y_max)

        # 2) If legal now → done
        if self.is_point_legal(context, cx, cy):
            return np.array([cx, cy], dtype=float)

        # 3) Push out of defense zone if needed
        if self.defense_x is not None and self.defense_y_half is not None:
            cy = np.sign(cy) * self.defense_y_half if cy != 0 else self.defense_y_half

        return np.array([cx, cy], dtype=float)

    """

def ball_is_striker_legal(context: RobotContext, x: float = None, y: float = None) -> bool:
    bx, by = context.ball.position[:2]

    if x is not None and y is not None:
        bx, by = x, y

    # striker zone
    in_zone = (
        STRIKER_X_MIN <= bx <= STRIKER_X_MAX and
        STRIKER_Y_MIN <= by <= STRIKER_Y_MAX
    )
    goal_x = -3400.0 if context.is_yellow else 3400.0
    in_defense = (
        (bx >= goal_x if context.is_yellow else bx <= goal_x)
        and abs(by) <= 1100.0
    )

    return in_zone and not in_defense

def project_to_striker_legal(context: RobotContext, x: float, y: float) -> np.ndarray:
   
    Projects a point into the nearest legal striker position.
   

    # Clamp to striker rectangle
    cx = np.clip(x, STRIKER_X_MIN, STRIKER_X_MAX)
    cy = np.clip(y, STRIKER_Y_MIN, STRIKER_Y_MAX)

    # If legal after clamp → done
    if ball_is_striker_legal(context, cx, cy):
        return np.array([cx, cy])

    # Otherwise, push out of defense area along Y
    defense_y = 1100.0
    cy = np.sign(cy) * defense_y if cy != 0 else defense_y

    return np.array([cx, cy])
"""