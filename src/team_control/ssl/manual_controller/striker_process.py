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
from team_control.ssl.rl.env import get_my_normalized_state


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

    # Load the optimizer state (momentum, etc.)
    vx, vy, w, kick, dribble, value = 0, 0, 0, 0, 0, 0
    action = vx, vy, w, kick, dribble
    while time.time() - start_time < 600: # 10 min
        # Update world model when new frame is available
        # while time < 1 and mode == testing
        # action = model.forward(torch.tensor(state, dtype=torch.float32))
        # a, b, c, d, e = action.detach()[0]
        # vx, vy, w, kick, dribble = float(a), float(b), float(c), float(d), int(e)
        if version < wm.get_version():
            version = wm.get_version()
            try:
                if robot_obj is not None:
                    robot_state, ball_state, teammates_state, opponents_state = get_my_normalized_state(robot_obj, ball_obj, teammates_obj, opponents_obj, is_team_yellow)
                    data_writer.write(robot_state, ball_state, teammates_state, opponents_state, action)
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

        distance_to_ball = math.hypot(bx - rx, by - ry)

        if distance_to_ball > 110:
            # Move to ball
            # print("we should move to ball")
            value = -5 * distance_to_ball / 2000
            vx, vy = robot_movement.go_to(robot_pos, ball_pos, kicker_to_target=True, stop_threshold=5)
            w = robot_movement.face_towards(robot_pos=robot_pos, target=ball_pos, max_angular_acceleration=3)
            kick, dribble = 0, 0
        else:
            # Align to goal
            y_goal = random.randint(-400, 400)
            goal_dir = math.atan2(y_goal - by, opp_goal_line - bx)
            angle_error = (goal_dir - rtheta + math.pi) % (2 * math.pi) - math.pi

            if abs(math.degrees(angle_error)) > 4:
                omega = 3.0 * angle_error
                vx, vy, w, kick, dribble = 0, 0, omega, 0, 1
                value = 1
            else:
                # print("here")
                if (rx-opp_goal_line <= 2000 and is_team_yellow) or (opp_goal_line-rx <= 2000 and not is_team_yellow):
                    kick = max(10 * abs(rx-opp_goal_line) / 2000, 3)
                    value = 5
                else:
                    kick = 1
                    value = 2
                # Kick the ball
                vx, vy, w, dribble = 0, 0, 0, 0

        cmd = RobotCommand(robot_id=robot_id, vx=vx, vy=vy, w=w, kick=kick, dribble=dribble)
        action = vx, vy, w, kick, dribble
        sender.send_command(cmd)

        # ---- CHECK FOR GOAL ----
        # Check if the ball crossed the goal line and is within the goal posts
        goal_y_min = -spawner.field.goal_width / 2
        goal_y_max = spawner.field.goal_width / 2

        if ball_pos[0] * opp_goal_line >= 0 and abs(ball_pos[0]) >= abs(opp_goal_line) and goal_y_min <= ball_pos[1] <= goal_y_max:
            goal_count += 1
            print(f"Total goals so far: {goal_count}")
            # Respawn the ball immediately after a goal
            # x, y = spawner.spawn_next_idle_ball(robot_pos)
            theta = 2 * math.pi * random.random()
            r = 500 * math.sqrt(random.random())

            x = r * math.cos(theta)
            y = r * math.sin(theta)

            ball_sender.send_ball_to_grsim(x, y, 0, 0)
            ball_pos = (x, y)

        # ---- BALL RESPAWNING LOGIC (DO NOT CHANGE) ----
        elif not spawner.is_ball_in_field(ball_pos):
            # x, y = spawner.spawn_next_idle_ball(robot_pos)
            theta = 2 * math.pi * random.random()
            r = 500 * math.sqrt(random.random())

            x = r * math.cos(theta)
            y = r * math.sin(theta)
            ball_sender.send_ball_to_grsim(x, y, 0, 0)
            ball_pos = (x, y)

    data_writer.close()
