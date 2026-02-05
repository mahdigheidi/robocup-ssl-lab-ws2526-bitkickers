"""
striker_process.py

Process for shooting at the goal.
"""
import math
import time
import random
import traceback

from team_control.network.sender import GrSimSender
from team_control.network.robot_command import RobotCommand
from team_control.network.send_ball_grsim import BallSender
from team_control.network.ball_spawner import BallSpawner
from team_control.robot.movement import RobotMovement
from team_control.world.model import WorldModel
# from team_control.ssl.rl.encoder import Encoder
from team_control.ssl.datasets.data_generator import DataWriter
# from team_control.ssl.rl.env import get_my_normalized_state


def striker_process(wm: WorldModel, division_name, sim_ip, cmd_listen_port, robot_id, is_team_yellow) -> None:
    # Initialize grSim command sender
    sender = GrSimSender(ip=sim_ip, port=cmd_listen_port, is_yellow=is_team_yellow)
    data_writer = DataWriter(f"{'yellow' if is_team_yellow else 'blue'}_striker")

    # Track world model
    version = 0

    # Initialize BallSpawner
    spawner = BallSpawner(division_name=division_name)
    ball_sender = BallSender()

    robot_obj, ball_obj, teammates_obj, opponents_obj = wm.get_robot_context(robot_id, is_team_yellow)
    robot_pos = robot_obj.position
    ball_pos = ball_obj.position

    robot_movement = RobotMovement()

    # robot_encoder = Encoder(robot_obj, teammates_obj, opponents_obj, ball_obj)
    # state = robot_encoder.get_state()
    # print("State: ", state)

    # Track number of goals scored
    goal_count = 0

    opp_goal_line = spawner.field.x_min if is_team_yellow else spawner.field.x_max

    # Initial ball spawn
    x, y = spawner.spawn_next_idle_ball(robot_pos)
    ball_sender.send_ball_to_grsim(x, y, 0, 0)
    start_time = time.time()
    # Timer for 120 seconds (you may change this for development)

    # csv_file = open('striker_data.csv','a')
    # model = MLP()
    # checkpoint = torch.load("striker_model.pt")

    # # Load the weights into the model
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.eval()
    areas = [(1000, 3500), (-1000, 1000), (-3500, -1000)]
    area = areas[robot_id - 1]
    # Load the optimizer state (momentum, etc.)
    vx, vy, w, kick, dribble = 0, 0, 0, 0, 0
    while time.time() - start_time < 600: # 10 min
        # Update world model when new frame is available
        # while time < 1 and mode == testing
        # action = model.forward(torch.tensor(state, dtype=torch.float32))
        # a, b, c, d, e = action.detach()[0]
        # vx, vy, w, kick, dribble = float(a), float(b), float(c), float(d), int(e)
        if version < wm.get_version():
            version = wm.get_version()
            try:
                # if robot_obj is not None:
                #     robot_state, ball_state, teammates_state, opponents_state = get_my_normalized_state(robot_obj, ball_obj, teammates_obj, opponents_obj, is_team_yellow)
                #     data_writer.write(robot_state, ball_state, teammates_state, opponents_state, action)
                robot_obj, ball_obj, teammates_obj, opponents_obj = wm.get_robot_context(robot_id, is_team_yellow)
                robot_pos = robot_obj.position
                ball_pos = ball_obj.position
                # robot, teammates, opponents, ball = robot_encoder.update_world_data(
                #     robot_obj, teammates_obj, opponents_obj, ball_obj)
                # print(vx, vy, w, kick, dribble)
                # print(state.shape)
                # state_help = [round(float(x), 5) for x in state]
                # state_str = '_'.join(map(str, state_help))
                # csv_file.write(f"{state_str},{float(vx)},{float(vy)},{float(w)},{float(kick)},{int(dribble)},{float(value)}\n")
                # print(state, vx, vy, w, kick, dribble)
                # state = robot_encoder.get_state()

            except Exception as e:
                robot_pos = 0, 0, 0
                ball_pos = 0, 0
                print(e)
                traceback.print_exc()

        rx, ry, rtheta = robot_pos
        bx, by = ball_pos

        if not area[0] <= rx <= area[1]:
                # dist[0] += 1
                detection = 0
                # print(case_id)
                # print("Robot not in area: ", robot_id)
                vx, vy = robot_movement.go_to(robot_pos, ((area[0]+area[1])/2, 0), kicker_to_target=True, stop_threshold=5)
                w = robot_movement.face_towards(robot_pos=robot_pos, target=(bx, by), max_angular_acceleration=3)
                kick, dribble = 0, 0
        else:
            # print("Robot in area: ", robot_id)
            if not area[0] <= bx <= area[1]:
                # dist[1] += 1
                detection = 1
                vx, vy, w, kick, dribble = 0, 0, 0, 0, 0
            else:

                distance_to_ball = math.hypot(bx - rx, by - ry)

                if distance_to_ball > 110:
                    # dist[2] += 1
                    detection = 2
                    vx, vy = robot_movement.go_to(robot_pos, (bx, by), kicker_to_target=True, max_acceleration=700)
                    w = robot_movement.face_towards(robot_pos=robot_pos, target=(bx, by), max_angular_acceleration=3)
                    kick, dribble = 0, 0
                else:
                    yellow_2 = wm.get_latest_frame().robots_yellow[2].position
                    yellow_3 = wm.get_latest_frame().robots_yellow[3].position
                    # Align to target
                    if robot_id == 1:  # target is the robot 2
                        x_target, y_target = yellow_2[0], yellow_2[1]
                        # print("Robot 1's target: ", x_target, y_target)
                    elif robot_id == 2:  # target is the robot 3
                        x_target, y_target = yellow_3[0], yellow_3[1]
                        # print(x_target, y_target)
                        # print("Robot 2's target: ", x_target, y_target)
                    else:  # robot_id == 3, target is the opponent goal center
                        x_target = opp_goal_line
                        y_target = 0
                        # print("Robot 3's target: ", x_target, y_target)
                    goal_dir = math.atan2(y_target - by, x_target - bx)
                    angle_error = (goal_dir - rtheta + math.pi) % (2 * math.pi) - math.pi

                    if abs(math.degrees(angle_error)) > 4:
                        # rotate to goal
                        # dist[3] += 1
                        detection = 3
                        omega = 3.0 * angle_error
                        vx, vy, w, kick, dribble = 0, 0, omega, 0, 1
                    else:
                        # Kick the ball
                        # dist[4] += 1
                        detection = 4
                        distance_to_target = math.hypot(x_target - bx, y_target - by) / 700
                        kick = distance_to_target if robot_id != 3 else 10
                        vx, vy, w, dribble = 0, 0, 0, 1
        cmd = RobotCommand(robot_id, vx, vy, w, kick, dribble)
        sender.send_command(cmd)

        # ---- CHECK FOR GOAL ----
        # Check if the ball crossed the goal line and is within the goal posts
        # goal_y_min = -spawner.field.goal_width / 2
        # goal_y_max = spawner.field.goal_width / 2

        # if ball_pos[0] * opp_goal_line >= 0 and abs(ball_pos[0]) >= abs(opp_goal_line) and goal_y_min <= ball_pos[1] <= goal_y_max:
        #     goal_count += 1
        #     print(f"Total goals so far: {goal_count}")
        #     # Respawn the ball immediately after a goal
        #     # x, y = spawner.spawn_next_idle_ball(robot_pos)
        #     theta = 2 * math.pi * random.random()
        #     r = 500 * math.sqrt(random.random())

        #     x = r * math.cos(theta)
        #     y = r * math.sin(theta)

        #     ball_sender.send_ball_to_grsim(x, y, 0, 0)
        #     ball_pos = (x, y)

        # # ---- BALL RESPAWNING LOGIC (DO NOT CHANGE) ----
        # elif not spawner.is_ball_in_field(ball_pos):
        #     # x, y = spawner.spawn_next_idle_ball(robot_pos)
        #     theta = 2 * math.pi * random.random()
        #     r = 500 * math.sqrt(random.random())

        #     x = r * math.cos(theta)
        #     y = r * math.sin(theta)
        #     ball_sender.send_ball_to_grsim(x, y, 0, 0)
        #     ball_pos = (x, y)

    # data_writer.close()
