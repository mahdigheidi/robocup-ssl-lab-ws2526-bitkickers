import math
import random
import time

from team_control.network.send_ball_grsim import BallSender
from team_control.robot.spawn_robot import send_robot_to_grsim
from team_control.network.robot_command import RobotCommand
from team_control.network.sender import GrSimSender
from team_control.ssl.rl.field import Field


def visualize_the_scenario(robot, ball, teammates, opponents, action):
    ball_sender = BallSender()
    send_robot_to_grsim(True, 0, robot[0]*4500, robot[1]*3000, math.degrees(robot[2]))
    if opponents is not None:
        for opponent in opponents:
            send_robot_to_grsim(False, 0, opponent[0]*4500, opponent[1]*3000)
    if teammates is not None:
        for teammate in teammates:
            send_robot_to_grsim(True, 1, teammate[0]*4500, teammate[1]*3000)
    ball_sender.send_ball_to_grsim(ball[0] * 4500, ball[1]*3000, ball[2], ball[3])
    cmd = RobotCommand(robot_id=0, vx=action[0], vy=action[1], w=action[2], kick=action[3], dribble=action[4])
    sender = GrSimSender()
    for _ in range(500000):
        sender.send_command(cmd)
    time.sleep(1)


def go_to_ball(field, opponent_goalie):
    x_max, y_max = field.x_max - field.penalty_depth, field.y_max
    max_dist_field = 2 * math.hypot(y_max, x_max)
    x = random.randint(-x_max, x_max)
    y = random.randint(-y_max, y_max)
    o = random.randint(0, 360)
    x_ball, y_ball = random.randint(-x_max, x_max), random.randint(-y_max, y_max)
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

    robot = [
        x / field.x_max,
        y / field.y_max,
        math.sin(math.radians(o)),
        math.cos(math.radians(o)),
        0, 0, 0, 1
    ]
    ball = [x_ball / field.x_max, y_ball / field.y_max, 0, 0]
    teammates = None
    opponents = None
    if opponent_goalie:
        o = random.randint(0, 360)
        opponents = [[
            random.randint(-field.x_max, field.x_max),
            random.randint(-field.y_max, field.y_max),
            math.sin(math.radians(o)),
            math.cos(math.radians(o)),
            0, 0, 0, 1,
        ]]
    action = [scale * cos_o, scale * sin_o, 0, 0, 0]

    return robot, ball, teammates, opponents, action


def kick_the_ball_to_opp_goal(field, opponent_goalie):
    x_max, y_max = field.x_max - field.penalty_depth, field.y_max
    max_dist_field = math.hypot(2 * y_max, field.x_max)
    x_y = random.randint(-x_max, 0)
    y_y = random.randint(-y_max, y_max)
    o_y = random.randint(0, 360)
    x_goal, y_goal = -field.x_max, random.randint(-field.goal_width // 2, field.goal_width // 2)
    distance_to_goal = math.hypot(x_goal - x_y, y_goal - y_y) / max_dist_field
    kick_power = (distance_to_goal) * 10
    angle_to_goal = math.degrees(math.atan((y_goal - y_y) / (x_goal - x_y)))
    if y_y < 0:
        o_y = 180 - abs(angle_to_goal)
    else:
        o_y = 180 + abs(angle_to_goal)

    cos_o = math.cos(math.radians(o_y))
    sin_o = math.sin(math.radians(o_y))

    x_ball = x_y - math.cos(math.radians(angle_to_goal)) * 110
    y_ball = y_y - math.sin(math.radians(angle_to_goal)) * 110

    robot = [
        x_y / field.x_max,
        y_y / field.y_max,
        round(sin_o, 4),
        round(cos_o, 4),
        0, 0, 0, 1
    ]
    ball = [x_ball / field.x_max, y_ball / field.y_max, 0, 0]
    teammates = None
    opponents = None
    if opponent_goalie:
        o = random.randint(0, 360)
        opponents = [[
            random.randint(-field.x_max, field.x_max),
            random.randint(-field.y_max, field.y_max),
            math.sin(math.radians(o)),
            math.cos(math.radians(o)),
            0, 0, 0, 1,
        ]]
    action = [0, 0, 0, kick_power, 0]

    return robot, ball, teammates, opponents, action


def rotate_to_face_the_ball(field, opponent_goalie):
    x_y, y_y, o_y = random.randint(-field.x_max, field.x_max), random.uniform(-field.y_max, field.y_max), random.randint(0, 360)
    random_angle = random.uniform(0, 360)
    x_ball = x_y - math.cos(math.radians(random_angle)) * 115
    y_ball = y_y - math.sin(math.radians(random_angle)) * 115

    dy = y_ball - y_y
    dx = x_ball - x_y
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

    w = gamma - o_y
    if w < 0:
        w = 360 + w
    w = math.radians(w) - math.pi

    sin_o = math.sin(math.radians(o_y))
    cos_o = math.cos(math.radians(o_y))

    robot = [
        x_y / field.x_max,
        y_y / field.y_max,
        round(sin_o, 4),
        round(cos_o, 4),
        0, 0, 0, 1
    ]
    ball = [x_ball / field.x_max, y_ball / field.y_max, 0, 0]
    teammates = None
    opponents = None
    if opponent_goalie:
        o = random.randint(0, 360)
        opponents = [[
            random.randint(-field.x_max, field.x_max) / field.x_max,
            random.randint(-field.y_max, field.y_max) / field.y_max,
            math.sin(math.radians(o)),
            math.cos(math.radians(o)),
            0, 0, 0, 1,
        ]]
    action = [0, 0, w, 0, 0]

    return robot, ball, teammates, opponents, action


def rotate_with_ball_to_goal(field, opponent_goalie):
    field_x_max, field_y_max = field.x_max - field.penalty_depth, field.y_max
    x_y, y_y, o_y = random.randint(-field_x_max+1, field_x_max), random.randint(-field.y_max, field.y_max), random.randint(0, 360)
    x_ball = x_y + math.cos(math.radians(o_y)) * 110
    y_ball = y_y + math.sin(math.radians(o_y)) * 110

    angle_to_goal = math.degrees(math.atan((0 - y_y) / (-field_x_max - x_y)))
    if y_y < 0:
        angle_to_goal = 180 - abs(angle_to_goal)
    else:
        angle_to_goal = 180 + abs(angle_to_goal)

    w = math.radians(angle_to_goal - o_y)

    cos_o = math.cos(math.radians(o_y))
    sin_o = math.sin(math.radians(o_y))
    robot = [
        x_y / field_x_max,
        y_y / field_y_max,
        round(sin_o, 4),
        round(cos_o, 4),
        0, 0, 0, 1
    ]
    ball = [x_ball / field_x_max, y_ball / field_y_max, 0, 0]
    teammates = None
    opponents = None
    if opponent_goalie:
        o = random.randint(0, 360)
        opponents = [[
            random.randint(-field.x_max, field.x_max),
            random.randint(-field.y_max, field.y_max),
            math.sin(math.radians(o)),
            math.cos(math.radians(o)),
            0, 0, 0, 1,
        ]]
    action = [0, 0, w, 0, 1]

    return robot, ball, teammates, opponents, action


def come_back_to_field_from_outside_immediately(field, opponent_goalie):
    x, y, o = 0, 0, random.uniform(0, 360)
    while True:
        x = random.randint(-2*field.x_max, 2*field.x_max)
        y = random.randint(-2*field.y_max, 2*field.y_max)
        if x > field.x_max or x < -field.x_max or y > field.y_max or y < -field.y_max:
            break
    x_ball, y_ball = random.uniform(-field.x_max, field.x_max), random.uniform(-field.y_max, field.y_max)

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
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)

    d = math.hypot(x_ball - x, y_ball - y)
    scale = 1
    if d > 2:
        scale = 4
    elif d > 1:
        scale = 3
    elif d > 0.5:
        scale = 2

    robot = [
        x / field.x_max,
        y / field.y_max,
        math.sin(math.radians(o)),
        math.cos(math.radians(o)),
        0, 0, 0, 1]
    ball = [x_ball / field.x_max, y_ball / field.y_max, 0, 0]
    teammates = None
    opponents = None
    if opponent_goalie:
        o = random.randint(0, 360)
        opponents = [[
            random.randint(-field.x_max, field.x_max),
            random.randint(-field.y_max, field.y_max),
            math.sin(math.radians(o)),
            math.cos(math.radians(o)),
            0, 0, 0, 1,
        ]]
    action = [scale * cos_angle, scale * sin_angle, 0, 0, 0]

    return robot, ball, teammates, opponents, action


scenarios = [
    go_to_ball,  # opp goalie can be there or not
    kick_the_ball_to_opp_goal,  # opp goalie can be there or not
    rotate_to_face_the_ball,  # opp goalie can be there or not
    rotate_with_ball_to_goal,  # opp goalie can be there or not
    come_back_to_field_from_outside_immediately,
    # move_forward_with_ball,  # opp goalie can be there or not
]

if __name__ == '__main__':
    field = Field('B')
    for scenario in scenarios:
        for i in range(5):
            opponent_goalie = random.random() >= 0.5
            robot, ball, teammates, opponents, action = scenario(field, opponent_goalie)
            visualize_the_scenario(robot, ball, teammates, opponents, action)
