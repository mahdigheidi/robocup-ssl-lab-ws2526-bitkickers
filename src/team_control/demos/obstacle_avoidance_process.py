"""
Obstacle avoidance process for SSL RoboCup.

Usage:
    python obstacle_avoidance_process.py scenario_1
"""

import importlib
import sys
import time

from team_control.motion.movement import RobotMovement
from team_control.motion.obstacle_pre_processor import preprocess_best_path
from team_control.motion.obstacles.helpers import check_stop_conditions, distance
from team_control.network.robot_command import RobotCommand
from team_control.network.ssl_sockets import grSimSender
from team_control.sim.send_ball_grsim import send_ball_to_grsim
from team_control.sim.spawn_robot import (
    remove_robots_not_in_scenario,
    spawn_robots_at_initial_positions,
)

TIME_STEP = 0.05  # seconds
X_CELL_SIZE = 100  # mm
Y_CELL_SIZE = 100  # mm


def obstacle_avoidance_process(wm):
    # Load scenario from command line argument or default
    scenario_name = sys.argv[1] if len(sys.argv) > 1 else "scenario_1"
    print(f"Loading scenario: {scenario_name}")

    scenario_module = importlib.import_module(
        f"team_control.demos.obstacle_scenarios.{scenario_name}"
    )
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
    static_robots_pos = [static_rob["start"] for static_rob in static_robots.values()]
    dynamic_robots_pos = [dynamic_rob["start"] for dynamic_rob in dynamic_robots.values()]

    # initialize dicts
    dyn_robot_obj = {}
    dyn_robot_cur_pos = {}

    # get all active robots ids from grSim
    robot_movement = RobotMovement()
    robots_yellow = wm.get_our_robots(True)
    robots_blue = wm.get_our_robots(False)

    # Remove Robots that are not present in the scenario
    remove_robots_not_in_scenario(robots_yellow, robots_blue, static_robots, dynamic_robots)

    # Remove ball
    send_ball_to_grsim(-5, -5, 0, 0)

    # spawn robots at their starting position
    spawn_robots_at_initial_positions(main_agent, static_robots, dynamic_robots)

    # initialize robot positions
    robot_pos = (main_agent["start"][0], main_agent["start"][1], 0)
    for dyn_robot_id, dyn_robot_data in dynamic_robots.items():
        dyn_robot_cur_pos[dyn_robot_id] = (
            dyn_robot_data["start"][0],
            dyn_robot_data["start"][1],
            0,
        )

    dynamic_paths = [(data["start"], data["goal"]) for id, data in dynamic_robots.items()]
    path = preprocess_best_path(
        main_agent["start"], main_agent["goal"], static_robots_pos, dynamic_paths
    )

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
                        isYellow=False, robot_id=dyn_robot_id
                    )
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
            target_pos=main_agent["goal"],
            static_robots=static_robots_pos,
            dynamic_robots=dynamic_robots_pos,
            elapsed_time=time.time() - start_time,
            time_limit=TIME_LIMIT,
            is_yellow=IS_YELLOW,
        )
        if stop:
            print(message)
            break

        # Move dynamic agents to their goals
        dynamic_robots_velocities = []
        for dyn_rob_id, dyn_robot_data in dynamic_robots.items():
            vx, vy = RobotMovement.moveagent(
                robot_pos=dyn_robot_cur_pos[dyn_rob_id], target=dyn_robot_data["goal"]
            )
            dynamic_robots_velocities.append((vx, vy))
            cmd = RobotCommand(robot_id=dyn_rob_id, vx=vx, vy=vy, w=0.0, kick=0, dribble=0)

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
