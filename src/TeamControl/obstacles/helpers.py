from __future__ import annotations
import math
from shapely.geometry import Point, Polygon
from TeamControl.robot.context import RobotContext, WorldState, POSSESSION_THRESHOLD, RoleArea
import numpy as np

from dataclasses import dataclass
# ----------------------------------------------------
# ------------ Constants and Polygons ----------------
# ----------------------------------------------------
ROBOT_RADIUS = 88.5 # m

COLLISION_DISTANCE = ROBOT_RADIUS * 2 + 10

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

def defense_area_polygon_for_team(is_yellow: bool):
    if is_yellow:
        return DEFENSE_AREA_POLYGON
    return [(-x, y) for (x, y) in DEFENSE_AREA_POLYGON]

# ----------------------------------------------------
# -------- STOP CONDITION CHECKING HELPERS -----------
# ----------------------------------------------------
def point_in_defense_area(context: RobotContext, agent_pos, robot_radius=ROBOT_RADIUS, allow=False):
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
    defense_area_poly = Polygon(defense_area_polygon_for_team(context.is_yellow))
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

def check_stop_conditions(agent_pos, target_pos, static_robots, dynamic_robots, elapsed_time=0, time_limit=30, is_goalie=False, is_yellow=True):
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

    if point_in_defense_area(agent_pos, allow=is_goalie,is_yellow=is_yellow):
        return True, "Robot entered opponent's defense area!\n0 POINTS"

    return False, ""

def ball_in_defense_area(context: RobotContext,ball_pos):
    return Point(ball_pos).within(Polygon(defense_area_polygon_for_team(context.is_yellow)))


def check_possession(
    ball_pos: np.ndarray, robot_pos: np.ndarray, threshold: float = POSSESSION_THRESHOLD
) -> bool:
    """Check if robot has possession of the ball based on distance threshold."""
    dist = float(np.linalg.norm(ball_pos - robot_pos[:2]))
    return dist < threshold


def avoid_collision_robust(rx, ry, tx, ty, opponents, safe_dist=COLLISION_DISTANCE + 20.0):
    """Avoid collisions with a dynamic list of opponent robots."""
    for op_pos in opponents:
        # op_pos is likely EntityState or np.ndarray
        pos = op_pos.position[:2] if hasattr(op_pos, "position") else op_pos[:2]
        d = math.hypot(tx - pos[0], ty - pos[1])
        if 5.0 < d < safe_dist:
            push = (safe_dist - d) / safe_dist
            dx = (tx - pos[0]) / d
            dy = (ty - pos[1]) / d
            tx += dx * push * (COLLISION_DISTANCE / 1.5)
            ty += dy * push * (COLLISION_DISTANCE / 1.5)
    return tx, ty


# ----------------------------------------------------
# ---------------- ROLE AREA SYSTEM ------------------
# ----------------------------------------------------
