"""
sandbox_process_ball_spawning.py

Example sandbox process for sending movement commands to a robot in grSim.

This script demonstrates how to:
    1. Initialize a grSim sender.
    2. Continuously read the world model (wm) for robot and ball positions.
    3. Spawn balls and respawn them automatically, tracking goals.
"""

import time

from team_control.motion.movement import RobotMovement
from team_control.network.robot_command import RobotCommand
from team_control.network.ssl_sockets import grSimSender
from team_control.sim.ball_spawner import BallSpawner

# --------------------------
# Ball spawner parameters
# --------------------------
DIVISION = "B"  # 'A' or 'B'
BALL_SPEED = 5.0  # m/s
GOAL_PROBABILITY = 0.75  # probability next ball goes toward goal (0..1)


def goalie_process(wm) -> None:
    # grSim connection parameters
    SIM_IP = "127.0.0.1"
    CMD_LISTEN_PORT = 20011
    IS_YELLOW = True
    ROBOT_ID = 0

    # Initialize grSim command sender
    sender = grSimSender(ip=SIM_IP, port=CMD_LISTEN_PORT, is_yellow=IS_YELLOW)

    # Track world model
    version = 0
    robot_pos = (0, 0, 0)
    ball_pos = (0, 0)
    prev_ball_pos = (-1, -1)

    # Initialize BallSpawner
    spawner = BallSpawner(
        division=DIVISION, ball_speed=BALL_SPEED, goal_probability=GOAL_PROBABILITY
    )

    robot_movement = RobotMovement()

    # make the robot go to the center of the goal at the start
    vx, vy = robot_movement.go_to(robot_pos=robot_pos, target=(spawner.goal_line_x - 200, 0))
    cmd = RobotCommand(robot_id=ROBOT_ID, vx=vx, vy=vy, w=0, kick=0, dribble=0)
    sender.send_command(cmd)

    # Track number of goals scored
    goal_count = 0

    # Initial ball spawn
    # x, y, vx, vy = spawner.spawn_next_ball()
    # send_ball_to_grsim(x, y, vx, vy)

    t = time.time()

    while True:
        # Update world model when new frame is available
        if version < wm.get_version():
            # print(1 / (time.time() - t))
            t = time.time()
            version = wm.get_version()
            try:
                robot_obj = wm.get_yellow_robots(isYellow=IS_YELLOW, robot_id=ROBOT_ID)
                robot_pos = robot_obj.position
                prev_ball_pos = ball_pos
                ball_pos = wm.get_latest_frame().ball.position

            except Exception:
                robot_pos = spawner.goal_line_x, 0, 0
                prev_ball_pos = (-1, -1)
                ball_pos = 0, 0

        # ---- GOALKEEPER STRATEGY ----

        # Tunable parameters
        POSITION_FRACTION = 0.2  # 0 = at goal center, 1 = at ball (for non-threatening balls)
        SPEED_THRESHOLD = 0.5  # m/s - minimum speed to consider ball as "moving toward goal"
        GOAL_MARGIN = 100  # mm - margin beyond goal posts to consider trajectory threatening
        BACKWARD_SHIFT_FACTOR = 1  # Factor to shift interception point backward along trajectory
        MAX_ACCELERATION_BASE = 1500
        MAX_ACCELERATION_DANGER = 6000

        # Goal center position
        goal_center_x = spawner.goal_line_x
        goal_center_y = 0

        # Calculate ball velocity from position change
        ball_velocity_x = ball_pos[0] - prev_ball_pos[0]
        ball_velocity_y = ball_pos[1] - prev_ball_pos[1]
        ball_speed = (ball_velocity_x**2 + ball_velocity_y**2) ** 0.5

        # Determine if ball is moving toward goal
        is_moving_toward_goal = False

        if ball_pos[0] != prev_ball_pos[0] and ball_speed > SPEED_THRESHOLD:
            # Calculate where ball trajectory intersects the goal line
            slope = ball_velocity_y / ball_velocity_x
            # y = slope * (x - ball_x) + ball_y
            intersect_y = slope * (goal_center_x - ball_pos[0]) + ball_pos[1]

            # Check if intersection is within goal (with margin) and ball is moving toward goal
            goal_y_min = -spawner.goal_width / 2 - GOAL_MARGIN
            goal_y_max = spawner.goal_width / 2 + GOAL_MARGIN

            if goal_y_min <= intersect_y <= goal_y_max and ball_velocity_x > 0:
                is_moving_toward_goal = True

        # print(f"\r{is_moving_toward_goal}", end='')

        # Case (b): Ball is moving toward goal - intercept the trajectory
        if is_moving_toward_goal:
            # Ball trajectory line: parametric form
            # Point on line: P = ball_pos + t * ball_velocity
            # Find closest point on this line to goalkeeper's current position

            # Vector from ball to robot
            dx = robot_pos[0] - ball_pos[0]
            dy = robot_pos[1] - ball_pos[1]

            # Project onto ball velocity vector to find closest point parameter t
            dot_product = dx * ball_velocity_x + dy * ball_velocity_y
            velocity_squared = ball_velocity_x**2 + ball_velocity_y**2
            t = dot_product / velocity_squared if velocity_squared > 0 else 0

            # Ensure t is non-negative (don't go behind the ball)
            t = max(0, t)

            # Closest point on trajectory
            closest_x = ball_pos[0] + t * ball_velocity_x
            closest_y = ball_pos[1] + t * ball_velocity_y

            # Calculate distance from robot to closest point
            distance_to_line = (
                (robot_pos[0] - closest_x) ** 2 + (robot_pos[1] - closest_y) ** 2
            ) ** 0.5

            # Shift the interception point backward along trajectory proportional to distance
            # This gives the goalkeeper more time to reach the interception point
            shift_distance = distance_to_line * BACKWARD_SHIFT_FACTOR

            # Calculate backward shift in terms of parameter t
            velocity_magnitude = (ball_velocity_x**2 + ball_velocity_y**2) ** 0.5
            if velocity_magnitude > 0:
                t_shift = shift_distance / velocity_magnitude
                t = max(0, t - t_shift)

            # Final interception target
            target_x = ball_pos[0] + t * ball_velocity_x
            target_y = ball_pos[1] + t * ball_velocity_y

            max_acceleration = MAX_ACCELERATION_DANGER

        # Case (a): Ball not threatening - position between ball and goal
        else:
            # Calculate target position as a fraction along the line from goal center to ball
            target_x = goal_center_x + POSITION_FRACTION * (ball_pos[0] - goal_center_x)
            target_y = goal_center_y + POSITION_FRACTION * (ball_pos[1] - goal_center_y)

            # Clamp target_y to within goal posts to ensure goalkeeper stays in reasonable range
            goal_y_min = -spawner.goal_width / 2
            goal_y_max = spawner.goal_width / 2
            target_y = max(goal_y_min, min(goal_y_max, target_y))

            max_acceleration = MAX_ACCELERATION_BASE

        # Move to target position
        vx, vy = robot_movement.go_to(
            robot_pos=robot_pos, target=(target_x, target_y), max_acceleration=max_acceleration
        )
        w = robot_movement.face_towards(robot_pos=robot_pos, target=ball_pos)

        # Send command to robot
        cmd = RobotCommand(robot_id=ROBOT_ID, vx=vx, vy=vy, w=w, kick=1, dribble=0)
        sender.send_command(cmd)

        # ---- CHECK FOR GOAL ----
        # Check if the ball crossed the goal line and is within the goal posts
        # goal_y_min = -spawner.goal_width / 2
        # goal_y_max = spawner.goal_width / 2
        # if ball_pos[0] >= spawner.goal_line_x and goal_y_min <= ball_pos[1] <= goal_y_max:
        #     goal_count += 1
        #     print(f"Total goals so far: {goal_count}")
        #     # Respawn the ball immediately after a goal
        #     x, y, vx, vy = spawner.spawn_next_ball()
        #     send_ball_to_grsim(x, y, vx, vy)
        #     prev_ball_pos = ball_pos
        #     ball_pos = (x, y)

        # ---- BALL RESPAWNING LOGIC (DO NOT CHANGE) ----
        # elif BallSpawner.is_ball_idle(ball_pos, prev_ball_pos) or not spawner.is_ball_in_field(ball_pos):
        #     x, y, vx, vy = spawner.spawn_next_ball()
        #     send_ball_to_grsim(x, y, vx, vy)
        #     prev_ball_pos = ball_pos
        #     ball_pos = (x, y)