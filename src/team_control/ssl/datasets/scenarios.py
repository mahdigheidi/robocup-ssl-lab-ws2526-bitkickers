import math
import random
import time

from team_control.network.send_ball_grsim import BallSender
from team_control.robot.spawn_robot import send_robot_to_grsim
from team_control.network.robot_command import RobotCommand
from team_control.network.sender import GrSimSender
from team_control.ssl.rl.field import Field

Y_P1_X = (1500, 3500)
Y_P2_X = (-1500, 1500)
Y_P3_X = (-3500, -1500)


# def visualize_the_scenario(robot, ball, teammates, opponents, action):
#     ball_sender = BallSender()
#     send_robot_to_grsim(True, 0, robot[0]*4500, robot[1]*3000, math.degrees(robot[2]))
#     if opponents is not None:
#         for opponent in opponents:
#             send_robot_to_grsim(False, 0, opponent[0]*4500, opponent[1]*3000)
#     if teammates is not None:
#         for teammate in teammates:
#             send_robot_to_grsim(True, 1, teammate[0]*4500, teammate[1]*3000)
#     ball_sender.send_ball_to_grsim(ball[0] * 4500, ball[1]*3000, ball[2], ball[3])
#     cmd = RobotCommand(robot_id=0, vx=action[0], vy=action[1], w=action[2], kick=action[3], dribble=action[4])
#     sender = GrSimSender()
#     for _ in range(500000):
#         sender.send_command(cmd)
#     time.sleep(1)


def go_to_point(field):
    ego_id = random.choice(["p1", "p2", "p3"])
    if ego_id == "p1":
        ego_area = Y_P1_X
    elif ego_id == "p2":
        ego_area = Y_P2_X
    else:
        ego_area = Y_P3_X
    max_dist_field = 2 * math.hypot(field.y_max, field.x_max)
    x = random.randint(ego_area[0], ego_area[1])
    y = random.randint(-field.y_max, field.y_max)
    o = random.randint(0, 360)
    x_ball, y_ball = random.randint(ego_area[0], ego_area[1]), random.randint(field.y_min, field.y_max)
    dy = y_ball - y
    dx = x_ball - x
    gamma = 90
    if dx == 0:
        if dy > 0:
            gamma = 90
        elif dy < 0:
            gamma = 270
    elif dy == 0:
        if dx > 0:
            gamma = 0
        else:
            gamma = 180
    else:
        gamma = abs(math.degrees(math.atan(dy / dx)))

    if dx > 0:
        if dy < 0:
            gamma = 360 - gamma
    else:
        if dy > 0:
            gamma = 180 - gamma
        elif dy < 0:
            gamma = 180 + gamma

    angle = gamma - o
    if angle < 0:
        angle = 360 + angle

    angle = math.radians(angle)
    cos_o = math.cos(angle)
    sin_o = math.sin(angle)

    d = math.hypot(dx, dy) / max_dist_field
    scale = d * 5
    if d > 0.5:
        scale = 4
    elif d > 0.25:
        scale = 3
    elif d > 0.15:
        scale = 2

    yellow_goalie = (random.randint(field.x_max - field.penalty_depth, field.x_max), random.randint(-field.goal_width // 2, field.goal_width // 2))
    yellow_p1 = (random.randint(Y_P1_X[0], Y_P1_X[1]), random.randint(-field.y_max, field.y_max))
    yellow_p2 = (random.randint(Y_P2_X[0], Y_P2_X[1]), random.randint(-field.y_max, field.y_max))
    yellow_p3 = (random.randint(Y_P3_X[0], Y_P3_X[1]), random.randint(-field.y_max, field.y_max))
    blue_goalie = (random.randint(field.x_min, field.x_min + field.penalty_depth), random.randint(-field.goal_width // 2, field.goal_width // 2))
    blue_p1 = (random.randint(-field.x_max, field.x_max), random.randint(-field.y_max, field.y_max))
    blue_p2 = (random.randint(-field.x_max, field.x_max), random.randint(-field.y_max, field.y_max))
    blue_p3 = (random.randint(-field.x_max, field.x_max), random.randint(-field.y_max, field.y_max))

    if ego_id == "p1":
        yellow_p1 = (x, y)
    elif ego_id == "p2":
        yellow_p2 = (x, y)
    else:
        yellow_p3 = (x, y)

    obs = [
        x / field.x_max,
        y / field.y_max,
        math.sin(math.radians(o)),
        math.cos(math.radians(o)),
        (x - x_ball) / field.x_max,
        (y - y_ball) / field.y_max,
        (x - yellow_goalie[0]) / field.x_max,
        (y - yellow_goalie[1]) / field.y_max,
        (x - yellow_p1[0]) / field.x_max,
        (y - yellow_p1[1]) / field.y_max,
        (x - yellow_p2[0]) / field.x_max,
        (y - yellow_p2[1]) / field.y_max,
        (x - yellow_p3[0]) / field.x_max,
        (y - yellow_p3[1]) / field.y_max,
        (x - blue_goalie[0]) / field.x_max,
        (y - blue_goalie[1]) / field.y_max,
        (x - blue_p1[0]) / field.x_max,
        (y - blue_p1[1]) / field.y_max,
        (x - blue_p2[0]) / field.x_max,
        (y - blue_p2[1]) / field.y_max,
        (x - blue_p3[0]) / field.x_max,
        (y - blue_p3[1]) / field.y_max,
    ]

    action = [scale * cos_o, scale * sin_o, 0, 0, 0]

    return obs, action


def kick_the_ball_to_opp_goal(field):
    ego_id = random.choice(["p1", "p2", "p3"])
    if ego_id == "p1":
        ego_area = Y_P1_X
    elif ego_id == "p2":
        ego_area = Y_P2_X
    else:
        ego_area = Y_P3_X
    x, y = random.randint(ego_area[0], ego_area[1]), random.randint(-field.y_max, field.y_max)
    o = random.randint(0, 360)
    max_dist_field = 2 * math.hypot(field.y_max, field.x_max)
    x_goal, y_goal = -field.x_max, 0
    distance_to_goal = math.hypot(x_goal - x, y_goal - y) / max_dist_field
    kick_power = (distance_to_goal) * 10
    angle_to_goal = math.degrees(math.atan((y_goal - y) / (x_goal - x)))
    if y < 0:
        o = 180 - abs(angle_to_goal)
    else:
        o = 180 + abs(angle_to_goal)

    cos_o = math.cos(math.radians(o))
    sin_o = math.sin(math.radians(o))

    x_ball = x - math.cos(math.radians(angle_to_goal)) * 110
    y_ball = y - math.sin(math.radians(angle_to_goal)) * 110

    yellow_goalie = (random.randint(field.x_max - field.penalty_depth, field.x_max), random.randint(-field.goal_width // 2, field.goal_width // 2))
    yellow_p1 = (random.randint(Y_P1_X[0], Y_P1_X[1]), random.randint(-field.y_max, field.y_max))
    yellow_p2 = (random.randint(Y_P2_X[0], Y_P2_X[1]), random.randint(-field.y_max, field.y_max))
    yellow_p3 = (random.randint(Y_P3_X[0], Y_P3_X[1]), random.randint(-field.y_max, field.y_max))
    blue_goalie = (random.randint(field.x_min, field.x_min + field.penalty_depth), random.randint(-field.goal_width // 2, field.goal_width // 2))
    blue_p1 = (random.randint(-field.x_max, field.x_max), random.randint(-field.y_max, field.y_max))
    blue_p2 = (random.randint(-field.x_max, field.x_max), random.randint(-field.y_max, field.y_max))
    blue_p3 = (random.randint(-field.x_max, field.x_max), random.randint(-field.y_max, field.y_max))

    if ego_id == "p1":
        yellow_p1 = (x, y)
    elif ego_id == "p2":
        yellow_p2 = (x, y)
    else:
        yellow_p3 = (x, y)

    obs = [
        x / field.x_max,
        y / field.y_max,
        sin_o,
        cos_o,
        (x - x_ball) / field.x_max,
        (y - y_ball) / field.y_max,
        (x - yellow_goalie[0]) / field.x_max,
        (y - yellow_goalie[1]) / field.y_max,
        (x - yellow_p1[0]) / field.x_max,
        (y - yellow_p1[1]) / field.y_max,
        (x - yellow_p2[0]) / field.x_max,
        (y - yellow_p2[1]) / field.y_max,
        (x - yellow_p3[0]) / field.x_max,
        (y - yellow_p3[1]) / field.y_max,
        (x - blue_goalie[0]) / field.x_max,
        (y - blue_goalie[1]) / field.y_max,
        (x - blue_p1[0]) / field.x_max,
        (y - blue_p1[1]) / field.y_max,
        (x - blue_p2[0]) / field.x_max,
        (y - blue_p2[1]) / field.y_max,
        (x - blue_p3[0]) / field.x_max,
        (y - blue_p3[1]) / field.y_max,
    ]
    action = [0, 0, 0, kick_power, 0]

    return obs, action


def rotate_to_face_the_ball(field):
    ego_id = random.choice(["p1", "p2", "p3"])
    if ego_id == "p1":
        ego_area = Y_P1_X
    elif ego_id == "p2":
        ego_area = Y_P2_X
    else:
        ego_area = Y_P3_X
    x, y = random.randint(ego_area[0], ego_area[1]), random.randint(-field.y_max, field.y_max)
    o = random.randint(0, 360)
    random_angle = random.uniform(0, 360)
    x_ball = x - math.cos(math.radians(random_angle)) * 115
    y_ball = y - math.sin(math.radians(random_angle)) * 115

    dy = y_ball - x
    dx = x_ball - y
    gamma = 90
    if dx == 0:
        if dy > 0:
            gamma = 90
        elif dy < 0:
            gamma = 270
    elif dy == 0:
        if dx > 0:
            gamma = 0
        else:
            gamma = 180
    else:
        gamma = abs(math.degrees(math.atan(dy / dx)))

    if dx > 0:
        if dy < 0:
            gamma = 360 - gamma
    else:
        if dy > 0:
            gamma = 180 - gamma
        elif dy < 0:
            gamma = 180 + gamma

    w = gamma - y
    if w < 0:
        w = 360 + w
    w = math.radians(w) - math.pi

    sin_o = math.sin(math.radians(o))
    cos_o = math.cos(math.radians(o))

    yellow_goalie = (random.randint(field.x_max - field.penalty_depth, field.x_max), random.randint(-field.goal_width // 2, field.goal_width // 2))
    yellow_p1 = (random.randint(Y_P1_X[0], Y_P1_X[1]), random.randint(-field.y_max, field.y_max))
    yellow_p2 = (random.randint(Y_P2_X[0], Y_P2_X[1]), random.randint(-field.y_max, field.y_max))
    yellow_p3 = (random.randint(Y_P3_X[0], Y_P3_X[1]), random.randint(-field.y_max, field.y_max))
    blue_goalie = (random.randint(field.x_min, field.x_min + field.penalty_depth), random.randint(-field.goal_width // 2, field.goal_width // 2))
    blue_p1 = (random.randint(-field.x_max, field.x_max), random.randint(-field.y_max, field.y_max))
    blue_p2 = (random.randint(-field.x_max, field.x_max), random.randint(-field.y_max, field.y_max))
    blue_p3 = (random.randint(-field.x_max, field.x_max), random.randint(-field.y_max, field.y_max))

    if ego_id == "p1":
        yellow_p1 = (x, y)
    elif ego_id == "p2":
        yellow_p2 = (x, y)
    else:
        yellow_p3 = (x, y)

    obs = [
        x / field.x_max,
        y / field.y_max,
        sin_o,
        cos_o,
        (x - x_ball) / field.x_max,
        (y - y_ball) / field.y_max,
        (x - yellow_goalie[0]) / field.x_max,
        (y - yellow_goalie[1]) / field.y_max,
        (x - yellow_p1[0]) / field.x_max,
        (y - yellow_p1[1]) / field.y_max,
        (x - yellow_p2[0]) / field.x_max,
        (y - yellow_p2[1]) / field.y_max,
        (x - yellow_p3[0]) / field.x_max,
        (y - yellow_p3[1]) / field.y_max,
        (x - blue_goalie[0]) / field.x_max,
        (y - blue_goalie[1]) / field.y_max,
        (x - blue_p1[0]) / field.x_max,
        (y - blue_p1[1]) / field.y_max,
        (x - blue_p2[0]) / field.x_max,
        (y - blue_p2[1]) / field.y_max,
        (x - blue_p3[0]) / field.x_max,
        (y - blue_p3[1]) / field.y_max,
    ]
    action = [0, 0, w, 0, 0]

    return obs, action


def rotate_with_ball_to_goal(field):
    ego_id = random.choice(["p1", "p2", "p3"])
    if ego_id == "p1":
        ego_area = Y_P1_X
    elif ego_id == "p2":
        ego_area = Y_P2_X
    else:
        ego_area = Y_P3_X
    x, y = random.randint(ego_area[0], ego_area[1]), random.randint(-field.y_max, field.y_max)
    o = random.randint(0, 360)
    # Generating positions of other robots

    x_ball = x + math.cos(math.radians(o)) * 110
    y_ball = y + math.sin(math.radians(o)) * 110

    angle_to_goal = math.degrees(math.atan((0 - y) / (field.x_min - y)))
    if y < 0:
        angle_to_goal = 180 - abs(angle_to_goal)
    else:
        angle_to_goal = 180 + abs(angle_to_goal)

    w = math.radians(angle_to_goal - o)

    cos_o = math.cos(math.radians(o))
    sin_o = math.sin(math.radians(o))
    yellow_goalie = (random.randint(field.x_max - field.penalty_depth, field.x_max), random.randint(-field.goal_width // 2, field.goal_width // 2))
    yellow_p1 = (random.randint(Y_P1_X[0], Y_P1_X[1]), random.randint(-field.y_max, field.y_max))
    yellow_p2 = (random.randint(Y_P2_X[0], Y_P2_X[1]), random.randint(-field.y_max, field.y_max))
    yellow_p3 = (random.randint(Y_P3_X[0], Y_P3_X[1]), random.randint(-field.y_max, field.y_max))
    blue_goalie = (random.randint(field.x_min, field.x_min + field.penalty_depth), random.randint(-field.goal_width // 2, field.goal_width // 2))
    blue_p1 = (random.randint(-field.x_max, field.x_max), random.randint(-field.y_max, field.y_max))
    blue_p2 = (random.randint(-field.x_max, field.x_max), random.randint(-field.y_max, field.y_max))
    blue_p3 = (random.randint(-field.x_max, field.x_max), random.randint(-field.y_max, field.y_max))

    if ego_id == "p1":
        yellow_p1 = (x, y)
    elif ego_id == "p2":
        yellow_p2 = (x, y)
    else:
        yellow_p3 = (x, y)

    obs = [
        x / field.x_max,
        y / field.y_max,
        sin_o,
        cos_o,
        (x - x_ball) / field.x_max,
        (y - y_ball) / field.y_max,
        (x - yellow_goalie[0]) / field.x_max,
        (y - yellow_goalie[1]) / field.y_max,
        (x - yellow_p1[0]) / field.x_max,
        (y - yellow_p1[1]) / field.y_max,
        (x - yellow_p2[0]) / field.x_max,
        (y - yellow_p2[1]) / field.y_max,
        (x - yellow_p3[0]) / field.x_max,
        (y - yellow_p3[1]) / field.y_max,
        (x - blue_goalie[0]) / field.x_max,
        (y - blue_goalie[1]) / field.y_max,
        (x - blue_p1[0]) / field.x_max,
        (y - blue_p1[1]) / field.y_max,
        (x - blue_p2[0]) / field.x_max,
        (y - blue_p2[1]) / field.y_max,
        (x - blue_p3[0]) / field.x_max,
        (y - blue_p3[1]) / field.y_max,
    ]
    action = [0, 0, w, 0, 1]

    return obs, action


# def come_back_to_field_from_outside_immediately(field, opponent_goalie):
#     x, y, o = 0, 0, random.uniform(0, 360)
#     while True:
#         x = random.randint(-2*field.x_max, 2*field.x_max)
#         y = random.randint(-2*field.y_max, 2*field.y_max)
#         if x > field.x_max or x < -field.x_max or y > field.y_max or y < -field.y_max:
#             break
#     x_ball, y_ball = random.uniform(-field.x_max, field.x_max), random.uniform(-field.y_max, field.y_max)

#     dy = y_ball - y
#     dx = x_ball - x
#     gamma = 90
#     if dx == 0:
#         if dy > 0:
#             gamma = 90
#         elif dy < 0:
#             gamma = 270
#     elif dy == 0:
#         if dx > 0:
#             gamma = 0
#         else:
#             gamma = 180
#     else:
#         gamma = abs(math.degrees(math.atan(dy / dx)))

#     if dx > 0:
#         if dy < 0:
#             gamma = 360 - gamma
#     else:
#         if dy > 0:
#             gamma = 180 - gamma
#         elif dy < 0:
#             gamma = 180 + gamma

#     angle = gamma - o
#     if angle < 0:
#         angle = 360 + angle

#     angle = math.radians(angle)
#     cos_angle = math.cos(angle)
#     sin_angle = math.sin(angle)

#     d = math.hypot(x_ball - x, y_ball - y)
#     scale = 1
#     if d > 2:
#         scale = 4
#     elif d > 1:
#         scale = 3
#     elif d > 0.5:
#         scale = 2

#     robot = [
#         x / field.x_max,
#         y / field.y_max,
#         math.sin(math.radians(o)),
#         math.cos(math.radians(o)),
#         0, 0, 0, 1]
#     ball = [x_ball / field.x_max, y_ball / field.y_max, 0, 0]
#     action = [scale * cos_angle, scale * sin_angle, 0, 0, 0]

#     return observation, action


scenarios = [
    go_to_point,
    kick_the_ball_to_opp_goal,
    rotate_to_face_the_ball,
    rotate_with_ball_to_goal,
    # come_back_to_field_from_outside_immediately,
    # move_forward_with_ball,
]

if __name__ == '__main__':
    field = Field('B')
    for _ in range(100000):
        obs, act = go_to_point(field)
        print(obs, act)
        obs, act = kick_the_ball_to_opp_goal(field)
        print(obs, act)
        obs, act = rotate_to_face_the_ball(field)
        obs, act = rotate_with_ball_to_goal(field)
