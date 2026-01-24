"""
Obstacle avoidance process for SSL RoboCup.

Usage:
    python obstacle_avoidance_process.py scenario_1
"""

import time
import random
from team_control.network.sender import GrSimSender
from team_control.ssl.rl.auxiliary import check_stop_conditions, distance
from team_control.network.robot_command import RobotCommand
from team_control.robot.movement import RobotMovement
from team_control.robot.spawn_robot import spawn_robots_at_initial_positions, remove_robots_not_in_scenario
from team_control.network.send_ball_grsim import BallSender
from team_control.ssl.manual_controller.obstacle_pre_processor import preprocess_best_path

TIME_STEP = 0.05  # seconds
X_CELL_SIZE = 100  # mm
Y_CELL_SIZE = 100  # mm


def obstacle_avoidance_process(wm, division, sim_ip, cmd_listen_port, robot_id, is_team_yellow):
    main_agent = {'id': 0, 'start': (random.randint(-4000, 4000), random.randint(-2500, 2500)),
                  'goal': (random.randint(-3000, 3000), random.randint(-2000, 2000)), 'isYellow': True}

    # Static robots is a dictionary of robots (id, start and goal position)
    num_yellow_static = random.randint(0, 3)
    num_yellow_dynamic = 3 - num_yellow_static
    static_robots = {}
    for i in range(num_yellow_static):
        static_robots[i] = {'start': (random.randint(-4000, 4000), random.randint(-2500, 2500)), 'goal': None, 'isYellow': True}

    dynamic_robots = {}

    sender_us = GrSimSender(
        ip=sim_ip, port=cmd_listen_port, is_yellow=is_team_yellow)
    sender_them = GrSimSender(
        ip=sim_ip, port=cmd_listen_port, is_yellow=not is_team_yellow)
    ball_sender = BallSender()

    version = 0

    # Pre-processing

    start_time = time.time()
    TIME_LIMIT = 30  # seconds

    # initialiaze the positions of static and dynamic robots
    static_robots_pos = [static_rob['start']
                         for static_rob in static_robots.values()]
    dynamic_robots_pos = [dynamic_rob['start'] for dynamic_rob in
                          dynamic_robots.values()]

    # initialize dicts
    dyn_robot_obj = {}
    dyn_robot_cur_pos = {}

    # get all active robots ids from grSim
    robot_movement = RobotMovement()
    robots_yellow = wm.get_our_robots(True)
    robots_blue = wm.get_our_robots(False)

    # Remove Robots that are not present in the scenario
    remove_robots_not_in_scenario(
        robots_yellow, robots_blue, static_robots, dynamic_robots)

    # Remove ball
    ball_sender.send_ball_to_grsim(-5, -5, 0, 0)

    # spawn robots at their starting position
    spawn_robots_at_initial_positions(
        main_agent, static_robots, dynamic_robots)

    # initialize robot positions
    robot_pos = (main_agent['start'][0], main_agent['start'][1], 0)
    for dyn_robot_id, dyn_robot_data in dynamic_robots.items():
        dyn_robot_cur_pos[dyn_robot_id] = (
            dyn_robot_data['start'][0], dyn_robot_data['start'][1], 0)

    dynamic_paths = [(data['start'], data['goal'])
                     for id, data in dynamic_robots.items()]
    path = preprocess_best_path(
        main_agent['start'], main_agent['goal'], static_robots_pos, dynamic_paths)

    while time.time() - start_time < TIME_LIMIT:
        if version < wm.get_version():
            version = wm.get_version()
            try:
                robot_obj = wm.get_our_robots(
                    us=is_team_yellow, robot_id=robot_id)
                robot_pos = robot_obj.position  # (x, y, orientation)
                # initialize dynamic robots positions
                dynamic_robots_pos = []
                for dyn_robot_id in dynamic_robots:
                    dyn_robot_obj[dyn_robot_id] = wm.get_our_robots(us=False, robot_id=dyn_robot_id)
                    dyn_robot_cur_pos[dyn_robot_id] = dyn_robot_obj[dyn_robot_id].position
                    dynamic_robots_pos.append(
                        dyn_robot_cur_pos[dyn_robot_id][:2])
            except Exception as e:
                print(e)

        pos_2d = robot_pos[:2]

        # check for stop conditions:
        # - reached target
        # - left field
        # - collision
        # - in opponent defense area
        stop, message = check_stop_conditions(
            division,
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
            vx, vy = robot_movement.go_to(robot_pos=dyn_robot_cur_pos[dyn_rob_id],
                                          target=dyn_robot_data['goal'])
            dynamic_robots_velocities.append((vx, vy))
            cmd = RobotCommand(robot_id=dyn_rob_id, vx=vx, vy=vy, w=0.0, kick=0,
                               dribble=0)

            sender_them.send_command(cmd)

        # Define your obstacle avoidance here
        # You can access the current positions of static and dynamic robots
        # from the lists static_robots_pos and dynamic_robots_pos
        # static_robots_pos: list of (x,y) positions
        # dynamic_robots_pos: list of (x,y) positions
        # ...
        if len(path) > 1 and distance(pos_2d, path[0]) <= 100:
            path.pop(0)

        vx, vy = robot_movement.go_to(
            robot_pos=robot_pos, target=path[0], max_speed=800)
        cmd = RobotCommand(robot_id=robot_id, vx=vx,
                           vy=vy, w=0.0, kick=0, dribble=0)

        sender_us.send_command(cmd)
    else:
        print("Time limit reached.\n0 POINTS")
