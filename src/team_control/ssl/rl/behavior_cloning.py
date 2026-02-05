"""
striker_process.py

Process for shooting at the goal.
"""
import math
import time
import random
import traceback

import torch
import torch.nn as nn

from team_control.network.sender import GrSimSender
from team_control.network.robot_command import RobotCommand
from team_control.network.send_ball_grsim import BallSender
from team_control.network.ball_spawner import BallSpawner
from team_control.robot.movement import RobotMovement
from team_control.world.model import WorldModel
from team_control.ssl.rl.actor import Actor


def cloner_process(wm: WorldModel, division_name, sim_ip, cmd_listen_port, robot_id, is_team_yellow) -> None:
    # Initialize grSim command sender
    sender = GrSimSender(ip=sim_ip, port=cmd_listen_port, is_yellow=is_team_yellow)

    # Track world model
    version = 0
    robot = (0, 0, 0)
    ball = (0, 0)

    # Initial ball spawn
    start_time = time.time()
    model = StrikerPolicy(47)
    checkpoint = torch.load("imitation.pt")
    model.load_state_dict(checkpoint)
    model.eval()

    kick = 0
    distance_to_ball = float('inf')

    obs = torch.zeros(47)
    while time.time() - start_time < 600:  # 10 min
        if version < wm.get_version():
            version = wm.get_version()
            try:
                latest_frame = wm.get_latest_frame()
                ball = latest_frame.ball.position
                robot = latest_frame.robots_yellow[robot_id].position
                yellows = latest_frame.robots_yellow
                blues = latest_frame.robots_blue
                obs_list = [
                    robot_id,
                    float(robot[0] / 4500),
                    float(robot[1] / 3000),
                    math.sin(robot[2]),
                    math.cos(robot[2]),
                    float(ball[0] / 4500),
                    float(ball[1] / 3000),
                ]
                obs_list.append(0)
                obs_list.append(yellows[0].position[0] / 4500)
                obs_list.append(yellows[0].position[1] / 3000)
                obs_list.append(math.sin(yellows[0].position[2]))
                obs_list.append(math.cos(yellows[0].position[2]))
                for i in range(1, 4):
                    r = yellows[i].position
                    obs_list.append(i)
                    obs_list.append(float(r[0] / 4500))
                    obs_list.append(float(r[1] / 3000))
                    obs_list.append(math.sin(r[2]))
                    obs_list.append(math.cos(r[2]))

                obs_list.append(0)
                obs_list.append(blues[0].position[0] / 4500)
                obs_list.append(blues[0].position[1] / 3000)
                obs_list.append(math.sin(blues[0].position[2]))
                obs_list.append(math.cos(blues[0].position[2]))
                for i in range(1, 4):
                    r = blues[i].position
                    obs_list.append(i)
                    obs_list.append(float(r[0] / 4500))
                    obs_list.append(float(r[1] / 3000))
                    obs_list.append(math.sin(r[2]))
                    obs_list.append(math.cos(r[2]))
                obs = torch.tensor(obs_list)

            except Exception as e:
                robot = 0, 0, 0
                ball = 0, 0
                print(e)
                traceback.print_exc()
        # print(cont_action)
        # print(kick_action)
        # print(dribble_action)
        robot_movement = RobotMovement()
        areas = [(1000, 3500), (-1000, 1000), (-3500, -1000)]
        area = areas[robot_id - 1]
        distance_to_ball = math.hypot(robot[0]-ball[0], robot[1]-ball[1])
        if not area[0] <= ball[0] <= area[1]:
            print(robot_id, ball, area)
            cont_action = model(obs)
            vx, vy, w, _ = cont_action.detach()
        else:
            if distance_to_ball > 115:
                vx, vy = robot_movement.go_to(robot, ball, kicker_to_target=True, stop_threshold=5)
                w = robot_movement.face_towards(robot_pos=robot, target=ball, max_angular_acceleration=3)
                kick, dribble = 0, 0
            else:
                yellow_2 = wm.get_latest_frame().robots_yellow[2].position
                yellow_3 = wm.get_latest_frame().robots_yellow[3].position
                # Align to target
                if robot_id == 1:  # target is the robot 2
                    x_target, y_target = yellow_2[0], yellow_2[1]
                elif robot_id == 2:  # target is the robot 3
                    x_target, y_target = yellow_3[0], yellow_3[1]
                else:  # robot_id == 3, target is the opponent goal center
                    x_target = -4500
                    y_target = 0
                    # print("Robot 3's target: ", x_target, y_target)
                goal_dir = math.atan2(y_target - ball[1], x_target - ball[0])
                angle_error = (goal_dir - robot[2] + math.pi) % (2 * math.pi) - math.pi
                print(angle_error)

                if abs(math.degrees(angle_error)) > 4:
                    # rotate to target
                    omega = 3.0 * angle_error
                    vx, vy, w, kick, dribble = 0, 0, omega, 0, 1
                else:
                    # Kick the ball
                    distance_to_target = math.hypot(x_target - ball[0], y_target - ball[1]) / 700
                    kick = distance_to_target if robot_id != 3 else 10
                    vx, vy, w, dribble = 0, 0, 0, 1
        dribble = 1 if distance_to_ball <= 120 else 0
        cmd = RobotCommand(robot_id, vx, vy, w, kick, dribble)
        sender.send_command(cmd)



def create_observation_vector(robot_id, robot_pos, ball_pos, teammates_pos, opponents_pos):
    teammates_pos[robot_id] = robot_pos
    distance_to_ball = math.hypot(robot_pos[0]-ball_pos[0], robot_pos[1]-ball_pos[1])
    areas = [(1000, 3500), (-1000, 1000), (-3500, -1000)]
    area = areas[robot_id - 1]
    obs = [
        distance_to_ball,
        area[0] / 1000,
        area[1] / 1000,
        robot_id,
        robot_pos[0] / 1000,
        robot_pos[1] / 1000,
        math.sin(robot_pos[2]),
        math.cos(robot_pos[2]),
        ball_pos[0] / 1000,
        ball_pos[1] / 1000,
    ]
    for i in range(4):
        obs.extend([
            i,
            teammates_pos[i][0] / 1000,
            teammates_pos[i][1] / 1000,
            math.sin(teammates_pos[i][2]),
            math.cos(teammates_pos[i][2]),
        ])
    for i in range(4):
        obs.extend([
            i,
            opponents_pos[i][0] / 1000,
            opponents_pos[i][1] / 1000,
            math.sin(opponents_pos[i][2]),
            math.cos(opponents_pos[i][2]),
        ])
    return obs

if __name__ == '__main__':
    robot_movement = RobotMovement()
    case = random.choice([0, 1, 2, 3, 4])
    robot_id = random.choice([1, 2, 3])
    areas = [(1000, 3500), (-1000, 1000), (-3500, -1000)]
    area = areas[robot_id - 1]
       # randomly create the robots in field here instead of reading from grSim
    if case == 0:
        rx = random.choice([
            random.randint(-4500, area[0]),
            random.randint(area[1], 4500),
        ])
        ry = random.randint(-3000, 3000)
        rtheta = random.uniform(-math.pi, math.pi)
        robot_pos = (rx, ry, rtheta)
        bx, by = random.randint(-4500, 4500), random.randint(-3000, 3000)
        ball_pos = (bx, by)
        teammates = [(random.randint(-4500, 4500), random.randint(-3000, 3000),
                        random.uniform(-math.pi, math.pi)) for i in range(4)]
        opponents = [(random.randint(-4500, 4500), random.randint(-3000, 3000),
                        random.uniform(-math.pi, math.pi)) for i in range(4)]
    elif case == 1:
        rx = random.randint(area[0], area[1])
        ry = random.randint(-3000, 3000)
        rtheta = random.uniform(-math.pi, math.pi)
        robot_pos = (rx, ry, rtheta)
        bx = random.choice([
            random.randint(-4500, area[0]),
            random.randint(area[1], 4500),
        ])
        by = random.randint(-3000, 3000)
        ball_pos = (bx, by)
        teammates = [(random.randint(-4500, 4500), random.randint(-3000, 3000),
                        random.uniform(-math.pi, math.pi)) for i in range(4)]
        opponents = [(random.randint(-4500, 4500), random.randint(-3000, 3000),
                        random.uniform(-math.pi, math.pi)) for i in range(4)]
    elif case == 2:
        rx = random.randint(area[0], area[1])
        ry = random.randint(-3000, 3000)
        rtheta = random.uniform(-math.pi, math.pi)
        robot_pos = (rx, ry, rtheta)
        while True:
            bx = random.randint(area[0], area[1])
            by = random.randint(-3000, 3000)
            if math.hypot(rx-bx, ry-by) > 110:
                break
        ball_pos = (bx, by)
        teammates = [(random.randint(-4500, 4500), random.randint(-3000, 3000),
                        random.uniform(-math.pi, math.pi)) for i in range(4)]
        opponents = [(random.randint(-4500, 4500), random.randint(-3000, 3000),
                        random.uniform(-math.pi, math.pi)) for i in range(4)]
    elif case == 3:
        rx = random.randint(area[0], area[1])
        ry = random.randint(-3000, 3000)
        rtheta = random.uniform(-math.pi, math.pi)
        robot_pos = (rx, ry, rtheta)
        ball_theta = random.choice([
            rtheta + (i * math.pi) / 5 for i in range(10)
        ])
        bx = rx + 110*math.cos(ball_theta)
        by = ry + 110*math.sin(ball_theta)
        ball_pos = (bx, by)
        teammates = [(random.randint(-4500, 4500), random.randint(-3000, 3000),
                        random.uniform(-math.pi, math.pi)) for i in range(4)]
        opponents = [(random.randint(-4500, 4500), random.randint(-3000, 3000),
                        random.uniform(-math.pi, math.pi)) for i in range(4)]
    else:
        rx = random.randint(area[0], area[1])
        ry = random.randint(-3000, 3000)
        rtheta = random.uniform(-math.pi, math.pi)
        robot_pos = (rx, ry, rtheta)
        bx = rx + 110*math.cos(rtheta)
        by = ry + 110*math.sin(rtheta)
        ball_pos = (bx, by)
        teammates = [(random.randint(-4500, 4500), random.randint(-3000, 3000),
                        random.uniform(-math.pi, math.pi)) for i in range(4)]
        opponents = [(random.randint(-4500, 4500), random.randint(-3000, 3000),
                        random.uniform(-math.pi, math.pi)) for i in range(4)]

    observation = create_observation_vector(robot_id, robot_pos, ball_pos, teammates, opponents)

    vx, vy, w, kick, dribble = 0, 0, 0, 0, 0

    distance_to_ball = math.hypot(bx - rx, by - ry)
    if not area[0] < rx < area[1]:
        # case 0
        # best point in area to get a pass next time
        vx, vy = robot_movement.go_to(
            robot_pos, ((area[0]+area[1])/2, 0), kicker_to_target=True, stop_threshold=5)
        w = robot_movement.face_towards(robot_pos, ball_pos, max_angular_acceleration=3)
    else:
        if not area[0] < bx < area[1]:
            # case 1
            vx, vy, w, kick, dribble = 0, 0, 0, 0, 0
        else:
            if distance_to_ball > 110:
                # case 2
                # Move to ball
                vx, vy = robot_movement.go_to(robot_pos, ball_pos, kicker_to_target=True, stop_threshold=5)
                w = robot_movement.face_towards(
                    robot_pos=robot_pos, target=ball_pos, max_angular_acceleration=3)
                kick, dribble = 0, 0
            else:
                x_target, y_target = -1, -1
                # define target
                if robot_id == 3:
                    # clear shot to the goal
                    x_target, y_target = -4500, 0
                    # pass back to the midfielder
                elif robot_id == 2:
                    # pass to striker in a good point
                    x_target, y_target, _ = teammates[3]
                    # clear shot to the goal
                else:
                    # pass to the midfielder
                    x_target, y_target, _ = teammates[2]
                    # pass to the striker
                    # shoot to the goal
                # Align to target
                goal_dir = math.atan2(y_target - by, x_target - bx)
                angle_error = (goal_dir - rtheta + math.pi) % (2 * math.pi) - math.pi

                if abs(math.degrees(angle_error)) > 4:
                    # case 3
                    omega = 4.0 * angle_error
                    vx, vy, w, kick, dribble = 0, 0, omega, 0, 1
                else:
                    # case 4
                    # Kick the ball
                    # kick with the distance to the target
                    vx, vy, w, dribble = 0, 0, 0, 0
                    distance_to_target = math.hypot(rx-x_target, ry-y_target)
                    if abs(x_target) == 4500:
                        kick = 10
                    else:
                        kick = distance_to_target / 750
    print(vx, vy, w, kick, dribble)
    actor = Actor()
    saved_model = torch.load('imitation.pt')
    actor.load_state_dict(saved_model)
    action = actor(torch.tensor(observation))
    print(action)
