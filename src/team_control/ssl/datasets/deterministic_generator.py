# positional_data_generator.py

import math
import random
from team_control.ssl.rl.field import Field

# =========================
# Constants
# =========================
COLLISION_DIST = 173.0
MAX_KICK_POWER = 10.0

# Assigned x-regions (mm)
Y_P1_X = (1500, 3500)
Y_P2_X = (-1500, 1500)
Y_P3_X = (-3500, -1500)


# =========================
# Helper functions
# =========================
def normalize_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def random_pose(x_range, y_range):
    x = random.uniform(*x_range)
    y = random.uniform(*y_range)
    o = random.uniform(0, 2 * math.pi)
    return x, y, o


def distance(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)


# =========================
# Main generator
# =========================
def generate_sample(field: Field):
    # -------------------------
    # Generate yellow team
    # -------------------------
    yellow_goalie = random_pose(
        (field.x_max-field.penalty_depth, field.x_max), (-field.goal_width/2, field.goal_width/2)
    )
    yellow_p1 = random_pose(Y_P1_X, (-field.y_max, field.y_max))
    yellow_p2 = random_pose(Y_P2_X, (-field.y_max, field.y_max))
    yellow_p3 = random_pose(Y_P3_X, (-field.y_max, field.y_max))

    # -------------------------
    # Generate blue team
    # -------------------------
    blue_goalie = random_pose(
        (field.x_min, field.x_min+field.penalty_depth), (-field.goal_width/2, field.goal_width/2)
    )
    blue_p1 = random_pose((-field.x_max, field.x_max), (-field.y_max, field.y_max))
    blue_p2 = random_pose((-field.x_max, field.x_max), (-field.y_max, field.y_max))
    blue_p3 = random_pose((-field.x_max, field.x_max), (-field.y_max, field.y_max))

    # -------------------------
    # Ball
    # -------------------------
    ball_x = random.uniform(field.x_min + field.penalty_depth, field.x_max - field.penalty_depth)
    ball_y = random.uniform(-field.y_max, field.y_max)

    # -------------------------
    # Ego selection
    # -------------------------
    ego_id = random.choice(["p1", "p2", "p3"])
    if ego_id == "p1":
        ego = yellow_p1
        ego_area = Y_P1_X
    elif ego_id == "p2":
        ego = yellow_p2
        ego_area = Y_P2_X
    else:
        ego = yellow_p3
        ego_area = Y_P3_X

    ego_x, ego_y, ego_o = ego
    ego_sin = math.sin(ego_o)
    ego_cos = math.cos(ego_o)

    # -------------------------
    # Observation
    # -------------------------
    entities = [
        ("ball", (ball_x, ball_y)),
        ("yellow_goalie", yellow_goalie),
        ("yellow_p1", yellow_p1),
        ("yellow_p2", yellow_p2),
        ("yellow_p3", yellow_p3),
        ("blue_goalie", blue_goalie),
        ("blue_p1", blue_p1),
        ("blue_p2", blue_p2),
        ("blue_p3", blue_p3),
    ]

    observation = [
        ego_x / field.x_max,
        ego_y / field.y_max,
        ego_sin,
        ego_cos,
    ]

    for _, ent in entities:
        ex, ey = ent[0], ent[1]
        observation.append((ego_x - ex) / field.x_max)
        observation.append((ego_y - ey) / field.y_max)

    # -------------------------
    # Action logic
    # -------------------------
    vx = vy = w = kick = dribble = 0.0

    inside_area = ego_area[0] <= ego_x <= ego_area[1]
    ball_inside_area = ego_area[0] <= ball_x <= ego_area[1]

    ego_angle = math.atan2(ego_sin, ego_cos)

    # Case A: outside area → go to center of area
    if not inside_area:
        target_x = (ego_area[0] + ego_area[1]) / 2
        target_y = 0.0
        dx = target_x - ego_x
        dy = target_y - ego_y
        target_angle = math.atan2(dy, dx)

        vx = math.cos(target_angle)
        vy = math.sin(target_angle)
        w = normalize_angle(target_angle - ego_angle)

    # Case B: inside area
    else:
        # B1: ball not in area → go to the center of the area
        if not ball_inside_area:
            target_x = (ego_area[0] + ego_area[1]) / 2
            target_y = 0.0
            dx = target_x - ego_x
            dy = target_y - ego_y
            target_angle = math.atan2(dy, dx)

            vx = math.cos(target_angle)
            vy = math.sin(target_angle)
            w = normalize_angle(target_angle - ego_angle)

        # B2: ball in area
        else:
            dist_to_ball = distance(ego_x, ego_y, ball_x, ball_y)
            ball_angle = math.atan2(ball_y - ego_y, ball_x - ego_x)
            angle_diff = normalize_angle(ball_angle - ego_angle)

            owns_ball = (
                dist_to_ball < COLLISION_DIST
                and abs(angle_diff) < math.radians(30)
            )

            # B2.a: owns ball → align and shoot
            if owns_ball:
                goal_x, goal_y = -field.x_max, 0.0
                goal_angle = math.atan2(goal_y - ego_y, goal_x - ego_x)

                w = normalize_angle(goal_angle - ego_angle)
                kick = MAX_KICK_POWER
                dribble = 1.0

            # B2.b: does not own ball → go behind ball
            else:
                target_x = ball_x - math.cos(ball_angle) * (COLLISION_DIST + 20)
                target_y = ball_y - math.sin(ball_angle) * (COLLISION_DIST + 20)

                dx = target_x - ego_x
                dy = target_y - ego_y
                move_angle = math.atan2(dy, dx)

                vx = math.cos(move_angle)
                vy = math.sin(move_angle)
                w = normalize_angle(ball_angle - ego_angle)

    action = [vx, vy, w, kick, dribble]
    return observation, action


# =========================
# Debug run
# =========================
if __name__ == "__main__":
    field = Field("B")
    # doing a for loop to generate 100000 samples
    # and writing them to a file
    for _ in range(100000):
        obs, act = generate_sample(field)
        print(",".join([f"{x:.4f}" for x in obs + act]))
