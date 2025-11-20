"""
striker_process.py

Process for shooting at the goal.
"""

from TeamControl.network.ssl_sockets import grSimSender
from TeamControl.network.robot_command import RobotCommand
from TeamControl.ball.send_ball_grsim import send_ball_to_grsim
from TeamControl.ball.ball_idle_spawner import BallIdleSpawner
from TeamControl.robot.Movement import RobotMovement

# from TeamControl.robot.shooting import RobotShooting
# from TeamControl.robot.obstacle_avoidance import AvoidObstacle
import math
import time

# --------------------------
# Ball idle spawner parameters
# --------------------------
DIVISION = "B"  # 'A' or 'B'


def striker_process(wm) -> None:
    # grSim connection parameters
    SIM_IP = "127.0.0.1"
    CMD_LISTEN_PORT = 20011
    IS_YELLOW = False
    ROBOT_ID = 0
    seed = 100

    # Initialize grSim command sender
    sender = grSimSender(ip=SIM_IP, port=CMD_LISTEN_PORT, is_yellow=IS_YELLOW)

    # Track world model
    version = 0
    robot_pos = (0, 0, 0)
    ball_pos = (0, 0)
    prev_ball_pos = (-1, -1)

    # Initialize BallSpawner
    spawner = BallIdleSpawner(division=DIVISION)

    # Track number of goals scored
    goal_count = 0

    # Initial ball spawn
    x, y, vx, vy = spawner.spawn_next_ball(robot_pos)
    send_ball_to_grsim(x, y, vx, vy)
    start_time = time.time()
    # Timer for 120 seconds (you may change this for development)

    robot_movement = RobotMovement()

    while time.time() - start_time < 120:
        # Update world model when new frame is available
        # while time < 1 and mode == testing
        if version < wm.get_version():
            version = wm.get_version()
            try:
                robot_obj = wm.get_yellow_robots(isYellow=IS_YELLOW, robot_id=ROBOT_ID)
                robot_pos = robot_obj.position
                prev_ball_pos = ball_pos
                ball_pos = wm.get_latest_frame().ball.position

            except Exception:
                robot_pos = 0, 0, 0
                prev_ball_pos = (-1, -1)
                ball_pos = 0, 0

        # ---- STRIKER STRATEGY ----

        rx, ry, rtheta = robot_pos
        bx, by = ball_pos

        cmd = None
        distance_to_ball = math.hypot(bx - rx, by - ry)

        if distance_to_ball > 110:
            # Move to ball
            vx, vy = robot_movement.go_to(robot_pos, ball_pos, kicker_to_target=True, stop_threshold=5)
            w = robot_movement.face_towards(robot_pos=robot_pos, target=ball_pos)

            cmd = RobotCommand(robot_id=ROBOT_ID, vx=vx, vy=vy, w=w, kick=0, dribble=0)
        else:
            # Align to goal
            goal_dir = math.atan2(0.0 - by, spawner.goal_line_x - bx)
            angle_error = (goal_dir - rtheta + math.pi) % (2 * math.pi) - math.pi

            if abs(math.degrees(angle_error)) > 4:
                omega = 4.0 * angle_error
                cmd = RobotCommand(robot_id=ROBOT_ID, vx=0, vy=0, w=omega, kick=0, dribble=1)
            else:
                # Kick the ball
                cmd = RobotCommand(robot_id=ROBOT_ID, vx=0, vy=0, w=0, kick=1, dribble=0)

        sender.send_command(cmd)

        # ---- CHECK FOR GOAL ----
        # Check if the ball crossed the goal line and is within the goal posts
        goal_y_min = -spawner.goal_width / 2
        goal_y_max = spawner.goal_width / 2
        # `- 300` to undo the correction of goal_line_x
        if ball_pos[0] >= spawner.goal_line_x - 300 and goal_y_min <= ball_pos[1] <= goal_y_max:
            goal_count += 1
            print(f"Total goals so far: {goal_count}")
            # Respawn the ball immediately after a goal
            x, y, vx, vy = spawner.spawn_next_ball(robot_pos)
            send_ball_to_grsim(x, y, vx, vy)
            ball_pos = (x, y)

        # ---- BALL RESPAWNING LOGIC (DO NOT CHANGE) ----
        elif not spawner.is_ball_in_field(ball_pos):
            x, y, vx, vy = spawner.spawn_next_ball(robot_pos)
            send_ball_to_grsim(x, y, vx, vy)
            ball_pos = (x, y)
