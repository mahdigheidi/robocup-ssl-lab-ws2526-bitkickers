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


def apply_velocity_repulsion(
    rx: float, ry: float, 
    vx: float, vy: float, 
    other_robots: list, 
    repulsion_dist: float = 500.0,
    max_repulsion: float = 2000.0
) -> tuple[float, float]:
    """
    Apply velocity-based repulsion to avoid collisions with other robots.
    
    This directly modifies the velocity vector to push the robot away from
    nearby robots, regardless of where the target is. Essential for
    preventing collisions when multiple robots converge on the same point.
    
    Args:
        rx, ry: Robot's current position
        vx, vy: Robot's current velocity command
        other_robots: List of other robots (teammates + opponents)
        repulsion_dist: Distance at which repulsion starts
        max_repulsion: Maximum repulsion velocity to add
        
    Returns:
        Modified (vx, vy) velocity with repulsion applied
    """
    repulsion_vx = 0.0
    repulsion_vy = 0.0
    
    for robot in other_robots:
        # Extract position from EntityState or array
        pos = robot.position[:2] if hasattr(robot, "position") else robot[:2]
        
        dist = math.hypot(rx - pos[0], ry - pos[1])
        
        if dist < repulsion_dist and dist > 1e-6:
            # Compute repulsion strength - stronger when closer
            # At dist=0: strength=1, at dist=repulsion_dist: strength=0
            normalized = dist / repulsion_dist
            strength = (1.0 - normalized) ** 2  # Quadratic falloff
            
            # Extra strong when very close (potential collision)
            if dist < COLLISION_DISTANCE + 50:
                strength *= 3.0
            
            # Direction: away from robot
            dx = (rx - pos[0]) / dist
            dy = (ry - pos[1]) / dist
            
            repulsion_vx += dx * strength * max_repulsion
            repulsion_vy += dy * strength * max_repulsion
    
    # Add repulsion to velocity
    new_vx = vx + repulsion_vx
    new_vy = vy + repulsion_vy
    
    # Optionally cap total velocity to prevent extreme speeds
    speed = math.hypot(new_vx, new_vy)
    max_speed = 5000.0  # mm/s
    if speed > max_speed:
        scale = max_speed / speed
        new_vx *= scale
        new_vy *= scale
    
    return new_vx, new_vy


def avoid_collision_robust(rx, ry, tx, ty, opponents, safe_dist=COLLISION_DISTANCE + 100.0, ball_pos=None):
    """
    Avoid collisions with a dynamic list of opponent robots.
    
    This function provides both:
    1. Target adjustment - pushes target away from obstacles
    2. Reactive repulsion - if robot is currently too close to another robot,
       adds a repulsion force to move away immediately
    
    Args:
        rx, ry: Robot's current position
        tx, ty: Target position
        opponents: List of opponent robots (EntityState or position arrays)
        safe_dist: Minimum safe distance to maintain
        ball_pos: Optional ball position [x, y]. If provided, robots with the ball
                  between them won't trigger repulsion (they can't directly collide).
        
    Returns:
        Adjusted (tx, ty) target position
    """
    # Reactive repulsion distance - start reacting well before collision
    # ROBOT_RADIUS is ~88.5mm, so two robots touch at ~177mm
    # Start reacting at ~400mm to have time to avoid
    reactive_dist = 400.0
    
    # Accumulate repulsion forces from all nearby robots
    repulsion_x = 0.0
    repulsion_y = 0.0
    
    for op_pos in opponents:
        # Extract position from EntityState or array
        pos = op_pos.position[:2] if hasattr(op_pos, "position") else op_pos[:2]
        
        # === Check if ball is between robot and obstacle ===
        # If so, skip repulsion - they can't directly collide
        if ball_pos is not None:
            robot_to_obs = math.hypot(rx - pos[0], ry - pos[1])
            if robot_to_obs > 1e-6:
                robot_to_ball = math.hypot(rx - ball_pos[0], ry - ball_pos[1])
                ball_to_obs = math.hypot(ball_pos[0] - pos[0], ball_pos[1] - pos[1])
                
                # Ball is "between" if it's closer to both than they are to each other,
                # and roughly on the line (sum of distances is close to direct distance)
                if robot_to_ball < robot_to_obs and ball_to_obs < robot_to_obs:
                    # Check if ball is roughly on the line (within tolerance)
                    path_length = robot_to_ball + ball_to_obs
                    if path_length < robot_to_obs * 1.3:  # 30% tolerance for "on the line"
                        # Ball is between us - skip repulsion for this robot
                        continue
        
        # === REACTIVE REPULSION: Check robot's current position ===
        robot_to_obs = math.hypot(rx - pos[0], ry - pos[1])
        
        if robot_to_obs < reactive_dist and robot_to_obs > 1e-6:
            # Robot is too close - compute repulsion force
            # Strength increases as distance decreases
            normalized_dist = robot_to_obs / reactive_dist  # 0 to 1
            # Strong exponential falloff - very strong when close
            strength = (1.0 - normalized_dist) ** 2
            
            # Extra strong repulsion when actually colliding (< 200mm)
            if robot_to_obs < 200.0:
                strength *= 3.0
            
            # Direction: away from the obstacle
            dx = (rx - pos[0]) / robot_to_obs
            dy = (ry - pos[1]) / robot_to_obs
            
            # Accumulate repulsion - use large magnitude to override target attraction
            repulsion_x += dx * strength * 500.0
            repulsion_y += dy * strength * 500.0
        
        # === TARGET ADJUSTMENT: Push target away from obstacles ===
        target_to_obs = math.hypot(tx - pos[0], ty - pos[1])
        
        if 1e-6 < target_to_obs < safe_dist:
            push = (safe_dist - target_to_obs) / safe_dist
            dx = (tx - pos[0]) / target_to_obs
            dy = (ty - pos[1]) / target_to_obs
            # Push proportional to safe_dist to actually maintain the desired clearance
            tx += dx * push * safe_dist
            ty += dy * push * safe_dist
    
    # Apply accumulated repulsion to target
    tx += repulsion_x
    ty += repulsion_y
    
    return tx, ty


# ----------------------------------------------------
# ---------------- ROLE AREA SYSTEM ------------------
# ----------------------------------------------------
