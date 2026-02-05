"""
striker_process.py

Process for shooting at the goal.
"""
import math
import random
import time

from team_control.network.sender import GrSimSender
from team_control.network.robot_command import RobotCommand
from team_control.robot.movement import RobotMovement
from team_control.world.model import WorldModel


# --------------------------
# Ball idle spawner parameters
# --------------------------
DIVISION = "B"  # 'A' or 'B'


def get_point_score(point, teammates, opponents):
    closest_teammate_dist = float('inf')
    for teammate in teammates:
        x, y, _ = teammate.position
        if math.hypot(point[0]-x, point[1]-y) < closest_teammate_dist:
            closest_teammate_dist = math.hypot(point[0]-x, point[1]-y)

    closest_opponent_dist = float('inf')
    for opponent in opponents:
        x, y, _ = opponent.position
        if math.hypot(point[0]-x, point[1]-y) < closest_opponent_dist:
            closest_opponent_dist = math.hypot(point[0]-x, point[1]-y)
    return -closest_teammate_dist + closest_opponent_dist


def is_shot_to_point_clear(ball, target, teammates, opponents):
    """
    ball: (x, y)
    target: (x, y)
    teammates / opponents: iterable with .position = (x, y, theta)
    robot_radius: physical robot radius in mm
    safety_margin: extra clearance
    """

    robot_radius = 88.5
    safety_margin = 20
    bx, by = ball
    tx, ty, _ = target

    # Direction vector of the shot
    dx = tx - bx
    dy = ty - by
    seg_len_sq = dx*dx + dy*dy

    # Degenerate case
    if seg_len_sq == 0:
        return False

    block_radius_sq = robot_radius + safety_margin

    def blocks(px, py):
        # Project point onto segment
        t = ((px - bx)*dx + (py - by)*dy) / seg_len_sq
        t = max(0.0, min(1.0, t))

        # Closest point on the segment
        cx = bx + t * dx
        cy = by + t * dy

        # Distance squared to segment
        dist_sq = (px - cx)**2 + (py - cy)**2
        return dist_sq < block_radius_sq

    # Check all robots
    for robot in teammates:
        rx, ry, _ = robot.gen_position
        if blocks(rx, ry):
            return False

    for robot in opponents:
        rx, ry, _ = robot.gen_position
        if blocks(rx, ry):
            return False

    return True


def striker_process(wm: WorldModel, sim_ip, cmd_listen_port, robot_id, is_team_yellow) -> None:
    time.sleep(1)
    # grSim connection parameters
    seed = 100

    # Initialize grSim command sender
    sender = GrSimSender(ip=sim_ip, port=cmd_listen_port, is_yellow=is_team_yellow)

    # Track world model
    version = 0
    robot_pos = (0, 0, 0)
    ball_pos = (0, 0)
    prev_ball_pos = (-1, -1)

    areas = [(1000, 3500), (-1000, 1000), (-3500, -1000)]
    area = areas[robot_id - 1]
    # exit(0)

    start_time = int(time.time())
    decision = random.choice([0, 1, 2])
    # Timer for 120 seconds (you may change this for development)

    robot_movement = RobotMovement()

    while True:
        # Update world model when new frame is available
        # while time < 1 and mode == testing
        # print("version: ", version)
        if time.time() - start_time > 3:
            decision = (decision + 1) % 3
            start_time = time.time()
        # print("world model version: ", wm.get_version())
        if version < wm.get_version():
            version = wm.get_version()
            try:
                robot_obj = wm.get_our_robots(us=is_team_yellow, robot_id=robot_id)
                robot_pos = robot_obj.gen_position
                # print(robot_id, robot_pos)
                prev_ball_pos = ball_pos
                ball_pos = wm.get_latest_frame().ball.gen_position(is_team_yellow)

            except Exception:
                robot_pos = 0, 0, 0
                prev_ball_pos = (-1, -1)
                ball_pos = 0, 0

        # ---- STRIKER STRATEGY ----

        rx, ry, rtheta = robot_pos
        bx, by = ball_pos
        prev_bx, prev_by = prev_ball_pos
        # if not is_team_yellow:
        #     print("Ball position: ", ball_pos)
        vx, vy, w, kick, dribble = 0, 0, 0, 0, 0

        cmd = None
        distance_to_ball = math.hypot(bx - rx, by - ry)

        teammates = wm.get_our_robots(us=is_team_yellow)
        opponents = wm.get_our_robots(us=not is_team_yellow)

        # if robot_id == 3:
        #     print(robot_pos)
        #     print(area)
        #     print(ball_pos)
        #     print("..............")

        if not area[0] < rx < area[1]:
            # best point in area to get a pass next time
            vx, vy = robot_movement.go_to(robot_pos, ((area[0]+area[1])/2, 0), kicker_to_target=True, stop_threshold=5)
            w = robot_movement.face_towards(robot_pos, ball_pos, max_angular_acceleration=3)
        else:
            if not area[0] < bx < area[1]:
                ball_vx, ball_vy = (bx - prev_bx)*12,  (by-prev_by)*12
                next_dist_to_ball = math.hypot(ball_vx+bx, ball_vy+by)
                if next_dist_to_ball <= distance_to_ball and distance_to_ball <= 1500:
                    vx, vy = robot_movement.go_to(robot_pos, ball_pos, kicker_to_target=True, stop_threshold=5)
                    w = robot_movement.face_towards(robot_pos=robot_pos, target=ball_pos, max_angular_acceleration=3)
                else:
                    # --- Sample candidate points inside the area ---
                    NUM_SAMPLES = 100
                    best_point = None
                    best_score = -float("inf")

                    for _ in range(NUM_SAMPLES):
                        px = random.uniform(area[0], area[1])
                        py = random.uniform(-2500, 2500)

                        # score based on distance to closest opponent
                        min_opponent_dist = float("inf")
                        for opp in opponents:
                            ox, oy, _ = opp.position
                            d = math.hypot(px - ox, py - oy)
                            if d < min_opponent_dist:
                                min_opponent_dist = d

                        if min_opponent_dist > best_score:
                            best_score = min_opponent_dist
                            best_point = (px, py)

                    # --- Move to the best point ---
                    if best_point is not None:
                        vx, vy = robot_movement.go_to(
                            robot_pos,
                            best_point,
                            kicker_to_target=False,
                            max_acceleration=500
                        )
                        w = robot_movement.face_towards(
                            robot_pos=robot_pos,
                            target=ball_pos,
                            max_angular_acceleration=3
                        )
                        kick, dribble = 0, 0
                    else:
                        vx, vy, w, kick, dribble = 0, 0, 0, 0, 0
            else:
                if distance_to_ball > 110:
                    # Move to ball
                    vx, vy = robot_movement.go_to(robot_pos, ball_pos, kicker_to_target=True, stop_threshold=5)
                    w = robot_movement.face_towards(robot_pos=robot_pos, target=ball_pos, max_angular_acceleration=3)
                    kick, dribble = 0, 0
                else:
                    x_target, y_target = -1, -1
                    # define target
                    if robot_id == 3:
                        # clear line of shot to the goal
                        if random.random() > 0.5 or decision == 0:
                            x_target, y_target = -4500, random.randint(-400, 400)
                        elif decision == 1:
                            x_target, y_target, _ = teammates[2].gen_position
                        elif decision == 2:
                            x_target, y_target, _ = teammates[1].gen_position
                        # rotate when holding the ball
                    elif robot_id == 2:
                        # pass to striker in a good point
                        if decision == 0:
                            x_target, y_target, _ = teammates[3].gen_position
                        elif decision == 1:
                            x_target, y_target = -4500, random.randint(-300, 300)
                        elif decision == 2:
                            x_target, y_target, _ = teammates[2].gen_position

                        # if pass to teammate3 is clear take the pass
                        # else, shoot to the goal

                    else:
                        # pass to the midfielder
                        if decision == 0:
                            x_target, y_target, _ = teammates[2].gen_position
                        elif decision == 1:
                            x_target, y_target, _ = teammates[3].gen_position
                        elif decision == 2:
                            x_target, y_target = -4500, random.randint(-200, 200)
                        # pass to the striker
                        # shoot to the goal
                    # Align to target
                    goal_dir = math.atan2(y_target - by, x_target - bx)
                    angle_error = (goal_dir - rtheta + math.pi) % (2 * math.pi) - math.pi

                    if abs(math.degrees(angle_error)) > 4:
                        omega = random.uniform(2, 4) * angle_error
                        vx, vy, w, kick, dribble = 0, 0, omega, 0, 1
                    else:
                        # Kick the ball
                        # kick with the distance to the target
                        vx, vy, w, dribble = 0, 0, 0, 0
                        distance_to_target = math.hypot(rx-x_target, ry-y_target)
                        if abs(x_target) == 4500:
                            kick = 10
                        else:
                            kick = distance_to_target / 800
                        # print(kick)
        # if not is_team_yellow:
        #     vx, vy = -vx, -vy
        cmd = RobotCommand(robot_id, round(vx, 1), round(vy, 1), w, kick, dribble)
        # if vx > 0:
        #     print("Vx:  ", round(vx, 1))
        # print(vx, vy, w, kick, dribble)
        sender.send_command(cmd)
