"""
striker_process.py

Process for shooting at the goal.
"""
import math
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from team_control.network.sender import GrSimSender
from team_control.network.robot_command import RobotCommand
from team_control.robot.movement import RobotMovement
from team_control.world.model import WorldModel
from team_control.ssl.rl.actor import Actor

# --------------------------
# Ball idle spawner parameters
# --------------------------
DIVISION = "B"  # 'A' or 'B'


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


def imitation():
    is_team_yellow = True  # for imitation we consider that we're playing on the right
    areas = [(1000, 3500), (-1000, 1000), (-3500, -1000)]

    robot_pos = (0, 0, 0)
    ball_pos = (0, 0)

    robot_movement = RobotMovement()

    observations, motions, kicks, dribbles = [], [], [], []
    dist = [0, 0, 0, 0, 0]

    for case in range(5):
        while True:
            if dist[case] == 10000:
                break
            detection = -1
            robot_id = random.choice([1, 2, 3])
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
                dist[0] += 1
                detection = 0
                # case 0
                # best point in area to get a pass next time
                vx, vy = robot_movement.go_to(
                    robot_pos, ((area[0]+area[1])/2, 0), kicker_to_target=True, stop_threshold=5)
                w = robot_movement.face_towards(robot_pos, ball_pos, max_angular_acceleration=3)
            else:
                if not area[0] < bx < area[1]:
                    # case 1
                    dist[1] += 1
                    detection = 1
                    vx, vy, w, kick, dribble = 0, 0, 0, 0, 0
                else:
                    if distance_to_ball > 110:
                        dist[2] += 1
                        detection = 2
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
                            dist[3] += 1
                            detection = 3
                            # case 3
                            omega = 4.0 * angle_error
                            vx, vy, w, kick, dribble = 0, 0, omega, 0, 1
                        else:
                            # case 4
                            # Kick the ball
                            # kick with the distance to the target
                            vx, vy, w, dribble = 0, 0, 0, 0
                            dist[4] += 1
                            detection = 4
                            distance_to_target = math.hypot(rx-x_target, ry-y_target)
                            kick = distance_to_target / 750
            if detection == case:
                observations.append(observation)
                motions.append([vx, vy, w])
                kicks.append([kick])
                dribbles.append([dribble])

    # print(dist)
    actor = Actor()
    optimizer = optim.Adam(actor.parameters(), lr=3e-4)

    # Convert dataset to tensors
    obs_tensor = torch.tensor(observations, dtype=torch.float32)
    motions_tensor = torch.tensor(motions, dtype=torch.float32)
    kicks_tensor = torch.tensor(kicks, dtype=torch.float32)
    dribbles_tensor = torch.tensor(dribbles, dtype=torch.float32)

    dataset = TensorDataset(obs_tensor, motions_tensor, kicks_tensor, dribbles_tensor)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    bce_loss = nn.BCEWithLogitsLoss()
    epochs = 5000
    for epoch in tqdm(range(epochs)):
        total_loss = 0.0

        for obs, motion, kick, dribble in dataloader:
            motion_pred, kick_pred, dribble_pred = actor(obs)

            loss = mse_loss(motion_pred, motion) + mse_loss(kick_pred, kick) + bce_loss(dribble_pred, dribble)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: avg loss = {avg_loss:.6f}")

    torch.save(actor.state_dict(), 'imitation.pt')


if __name__ == '__main__':
    imitation()
