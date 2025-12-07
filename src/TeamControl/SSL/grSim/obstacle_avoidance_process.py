"""
Obstacle avoidance process for SSL RoboCup.

Usage:
    python obstacle_avoidance_process.py scenario_1
"""

import sys
import time
import importlib
import random
import math
from shapely.geometry import Point
from TeamControl.network.ssl_sockets import grSimSender
from TeamControl.obstacles.helpers import *
from TeamControl.network.robot_command import RobotCommand
from TeamControl.robot.Movement import RobotMovement
from TeamControl.robot.spawn_robot import spawn_robots_at_initial_positions, remove_robots_not_in_scenario
from TeamControl.ball.send_ball_grsim import send_ball_to_grsim
from TeamControl.SSL.grSim.obstacle_pre_processor import preprocess_best_path

TIME_STEP = 0.05  # seconds
X_CELL_SIZE = 100  # mm
Y_CELL_SIZE = 100  # mm

def obstacle_avoidance_process(wm):
    # Load scenario from command line argument or default
    scenario_name = sys.argv[1] if len(sys.argv) > 1 else "scenario_1"
    print(f"Loading scenario: {scenario_name}")

    scenario_module = importlib.import_module(f"TeamControl.obstacles.{scenario_name}")
    main_agent, static_robots, dynamic_robots = scenario_module.get_scenario()

    # grSim connection setup
    SIM_IP = "127.0.0.1"
    CMD_LISTEN_PORT = 20011
    IS_YELLOW = True
    ROBOT_ID = 0

    sender_yellow = grSimSender(ip=SIM_IP, port=CMD_LISTEN_PORT, is_yellow=True)
    sender_blue = grSimSender(ip=SIM_IP, port=CMD_LISTEN_PORT, is_yellow=False)

    version = 0

    # Pre-processing

    start_time = time.time()
    TIME_LIMIT = 30  # seconds

    # initialiaze the positions of static and dynamic robots
    static_robots_pos = [static_rob['start'] for static_rob in static_robots.values()]
    dynamic_robots_pos = [dynamic_rob['start'] for dynamic_rob in
                         dynamic_robots.values()]

    # initialize dicts
    dyn_robot_obj = {}
    dyn_robot_cur_pos = {}

    # get all active robots ids from grSim
    robot_movement = RobotMovement()
    robots_yellow = wm.get_our_robots(True)
    robots_blue = wm.get_our_robots(False)
    # sender_yellow.send_command(RobotCommand(robot_id=ROBOT_ID, vx=-3, vy=-3, w=0.0, kick=0, dribble=0))

    # Remove Robots that are not present in the scenario
    remove_robots_not_in_scenario(robots_yellow, robots_blue, static_robots, dynamic_robots)

    # Remove ball
    send_ball_to_grsim(-5, -5, 0, 0)

    # spawn robots at their starting position
    spawn_robots_at_initial_positions(main_agent, static_robots, dynamic_robots)

    # initialize robot positions
    robot_pos = (main_agent['start'][0], main_agent['start'][1], 0)
    for dyn_robot_id, dyn_robot_data in dynamic_robots.items():
        dyn_robot_cur_pos[dyn_robot_id] = (dyn_robot_data['start'][0], dyn_robot_data['start'][1], 0)

    dynamic_paths = [(data['start'], data['goal']) for id, data in dynamic_robots.items()]
    path = preprocess_best_path(main_agent['start'], main_agent['goal'], static_robots_pos, dynamic_paths)

    while time.time() - start_time < TIME_LIMIT:
        if version < wm.get_version():
            version = wm.get_version()
            try:
                robot_obj = wm.get_yellow_robots(isYellow=IS_YELLOW, robot_id=ROBOT_ID)
                robot_pos = robot_obj.position  # (x, y, orientation)
                # initialize dynamic robots positions
                dynamic_robots_pos = []
                for dyn_robot_id in dynamic_robots:
                    dyn_robot_obj[dyn_robot_id] = wm.get_yellow_robots(
                        isYellow=False,
                        robot_id=dyn_robot_id)
                    dyn_robot_cur_pos[dyn_robot_id] = dyn_robot_obj[dyn_robot_id].position
                    dynamic_robots_pos.append(dyn_robot_cur_pos[dyn_robot_id][:2])
            except Exception:
                pass

        pos_2d = robot_pos[:2]

        # check for stop conditions:
        # - reached target
        # - left field
        # - collision
        # - in opponent defense area
        stop, message = check_stop_conditions(
            agent_pos=pos_2d,
            target_pos=main_agent['goal'],
            static_robots=static_robots_pos,
            dynamic_robots=dynamic_robots_pos,
            elapsed_time=time.time() - start_time,
            time_limit=TIME_LIMIT
        )
        if stop:
            print(message)
            break

        # Move dynamic agents to their goals
        dynamic_robots_velocities = []
        for dyn_rob_id, dyn_robot_data in dynamic_robots.items():
            vx, vy = RobotMovement.moveagent(robot_pos=dyn_robot_cur_pos[dyn_rob_id],
                                             target=dyn_robot_data['goal'])
            dynamic_robots_velocities.append((vx, vy))
            cmd = RobotCommand(robot_id=dyn_rob_id, vx=vx, vy=vy, w=0.0, kick=0,
                               dribble=0)

            sender_blue.send_command(cmd)

        # Define your obstacle avoidance here
        # You can access the current positions of static and dynamic robots
        # from the lists static_robots_pos and dynamic_robots_pos
        # static_robots_pos: list of (x,y) positions
        # dynamic_robots_pos: list of (x,y) positions 
        # ...
        if len(path) > 1 and distance(pos_2d, path[0]) <= 100:
            path.pop(0)

        vx, vy = robot_movement.go_to(robot_pos=robot_pos, target=path[0], max_acceleration=2000)
        cmd = RobotCommand(robot_id=ROBOT_ID, vx=vx, vy=vy, w=0.0, kick=0, dribble=0)

        sender_yellow.send_command(cmd)
    else:
        print("Time limit reached.\n0 POINTS")

def will_collide_in_next_frame(robot_pos, robot_velocity, dyn_obstacle_pos, dyn_obstacle_velocity):
    """
    Check if the robot will collide with a dynamic obstacle in the next frame.

    Args:
        robot_pos (tuple): (x, y) position of the robot.
        robot_velocity (tuple): (vx, vy) velocity of the robot.
        dyn_obstacle_pos (tuple): (x, y) position of the dynamic obstacle.
        dyn_obstacle_velocity (tuple): (vx, vy) velocity of the dynamic obstacle.
    Returns:
        bool: True if a collision will occur, False otherwise.
    """
    t_x = (robot_pos[0] - dyn_obstacle_pos[0]) / (dyn_obstacle_velocity[0] - robot_velocity[0])
    t_y = (robot_pos[1] - dyn_obstacle_pos[1]) / (dyn_obstacle_velocity[1] - robot_velocity[1])
    if abs(t_x - t_y) < TIME_STEP:
        # check if at that time they are close enough to collide
        return True
    return False

def direct_path_is_clear(start_pos, end_pos, static_obstacles, dynamic_obstacles, dynamic_obstacles_velocities, robot_movement: RobotMovement):
    """
    Function to check if the direct path between start_pos and end_pos
    is clear of obstacles and does not violate any conditions.

    Args:
        start_pos (tuple): (x, y) starting position.
        end_pos (tuple): (x, y) ending position.
        static_obstacles (list): List of (x, y) positions of static obstacles.
        dynamic_obstacles (list): List of (x, y) positions of dynamic obstacles.
    Returns:
        bool: True if path is clear, False otherwise.
    """
    # Here you would implement the actual checks for:
    # - opponent defense area crossing
    # the defence area has 4 lines, we can check if the line between start_pos and end_pos
    # intersects with any of these lines.
    if end_pos[0] == start_pos[0]:
        a = float('inf')
    else:
        a = (end_pos[1] - start_pos[1]) / (end_pos[0] - start_pos[0])
    b = start_pos[1] - a * start_pos[0]
    X1, X2 = DEFENSE_AREA_POLYGON[0][0] - 2 * ROBOT_RADIUS, DEFENSE_AREA_POLYGON[1][0] + 2 * ROBOT_RADIUS
    Y1, Y2 = DEFENSE_AREA_POLYGON[0][1] + 2 * ROBOT_RADIUS, DEFENSE_AREA_POLYGON[3][1] - 2 * ROBOT_RADIUS
    if a == float('inf'):
        if start_pos[1] <= Y1 + ROBOT_RADIUS and end_pos[1] >= Y2 or start_pos[1] >= Y2 and end_pos[1] <= Y1:
            return False
    elif a == 0:
        if start_pos[0] <= X1 and end_pos[0] >= X2 or start_pos[0] >= X2 and end_pos[0] <= X1:
            return False
    else:
        y_at_X1 = a * X1 + b
        y_at_X2 = a * X2 + b
        if (Y1 <= y_at_X1 <= Y2) or (Y1 <= y_at_X2 <= Y2):
            return False
        x_at_Y1 = (Y1 - b) / a
        x_at_Y2 = (Y2 - b) / a
        if (X1 <= x_at_Y1 <= X2) or (X1 <= x_at_Y2 <= X2):
            return False
    # - field boundaries
    # no need to check this, all generated points are inside the field

    # - proximity to static obstacles
    for obs in static_obstacles:
        if distance_to_line_segment(start_pos, end_pos, obs) < 2 * ROBOT_RADIUS:
            return False
    # - potential collisions with dynamic obstacles
    # we check here if in the next frame we will be too close to any dynamic obstacle
    # if we hit each other in the next frame, we consider the path not clear
    vx, vy = robot_movement.go_to(robot_pos=(start_pos[0], start_pos[1], 0), target=end_pos)
    for obs, vel in zip(dynamic_obstacles, dynamic_obstacles_velocities):
        t_x = (start_pos[0] - obs[0]) / (vel[0] - vx)
        t_y = (start_pos[1] - obs[1]) / (vel[1] - vy)
        if abs(t_x - t_y) < TIME_STEP:
            return False

    return True


def is_point_inside_field(point):
    """
    Check if a given point is inside the soccer field.

    Args:
        point (tuple): (x, y) coordinates.
    Returns:
        bool: True if inside the field polygon, False otherwise.
    """
    FIELD_POLYGON_WITH_MARGIN = Polygon([
        (-4350, -2750),  # bottom-left corner
        (-4350, 2750),   # top-left corner
        (4350, 2750),    # top-right corner
        (4350, -2750),   # bottom-right corner
    ])
    field_polygon = Polygon(FIELD_POLYGON_WITH_MARGIN)
    point_geom = Point(point)
    return field_polygon.contains(point_geom)

def distance_to_line_segment(start, end, point):
    """
    Calculate the minimum distance from a point to a line segment defined by start and end points.

    Args:
        start (tuple): (x, y) coordinates of the start point of the line segment.
        end (tuple): (x, y) coordinates of the end point of the line segment.
        point (tuple): (x, y) coordinates of the point.
    Returns:
        float: Minimum distance from the point to the line segment.
    """
    # Vector AB
    ABx = end[0] - start[0]
    ABy = end[1] - start[1]
    # Vector AP
    APx = point[0] - start[0]
    APy = point[1] - start[1]
    # Vector BP
    BPx = point[0] - end[0]
    BPy = point[1] - end[1]

    # Calculate the magnitude squared of AB
    magAB2 = ABx * ABx + ABy * ABy
    if magAB2 == 0:
        # A and B are the same point
        return math.hypot(APx, APy)

    # Calculate the projection of AP onto AB to find the parameter t
    t = (APx * ABx + APy * ABy) / magAB2

    if t < 0:
        # Closest to point A
        return math.hypot(APx, APy)

    if t > 1:
        # Closest to point B
        return math.hypot(BPx, BPy)

    # Projection falls on the segment
    closestX = start[0] + t * ABx
    closestY = start[1] + t * ABy
    dx = point[0] - closestX
    dy = point[1] - closestY
    return math.hypot(dx, dy)
