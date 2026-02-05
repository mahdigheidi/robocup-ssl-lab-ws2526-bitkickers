import math
import time
import traceback

from team_control.network.sender import GrSimSender
from team_control.network.robot_command import RobotCommand
from team_control.network.ball_spawner import BallSpawner
from team_control.robot.movement import RobotMovement
from team_control.ssl.datasets.data_generator import DataWriter
from team_control.world.model import WorldModel

# --------------------------
# Ball spawner parameters
# --------------------------
BALL_SPEED = 5.0  # m/s
GOAL_PROBABILITY = 0.75


def goalie_process(wm: WorldModel, sim_ip, cmd_listen_port, robot_id, is_team_yellow) -> None:
    time.sleep(1)
    # Initialize grSim command sender
    sender = GrSimSender(ip=sim_ip, port=cmd_listen_port, is_yellow=is_team_yellow)

    # Track world model
    version = 0
    robot_pos = (0, 0, 0)
    prev_robot_pos = (0, 0, 0)
    ball_pos = (0, 0)
    prev_ball_pos = (-1, -1)

    robot_movement = RobotMovement()

    # Track number of goals scored
    goal_center_y = 0

    # prev_vx, prev_vy, prev_w = None, None, None

    while True:
        # Update world model when new frame is available
        if version < wm.get_version():
            version = wm.get_version()
            try:
                # if prev_vx:
                #     robot_vel = [(robot_pos[i] - prev_robot_pos[i]) * 0.012 for i in range(len(robot_pos))]
                #     robot_vel[2] *= 1000
                #     ball_vel = [(ball_pos[i] - prev_ball_pos[i]) * 0.012 for i in range(len(ball_pos))]

                robot_pos = wm.get_our_robots(us=is_team_yellow, robot_id=robot_id).gen_position
                ball_pos = wm.get_latest_frame().ball.gen_position(is_team_yellow)
                prev_robot_pos = robot_pos
                prev_ball_pos = ball_pos

            except Exception as e:
                prev_ball_pos = (-1, -1)
                prev_robot_pos = (-1, -1, -1)
                ball_pos = 0, 0
                robot_pos = (0, 0)
                traceback.print_exc()
                print(e)

        # ---- GOALKEEPER STRATEGY ----

        # Tunable parameters
        # 0 = at goal center, 1 = at ball (for non-threatening balls)
        POSITION_FRACTION = 0.2
        SPEED_THRESHOLD = 0.5  # m/s - minimum speed to consider ball as "moving toward goal"
        GOAL_MARGIN = 100  # mm - margin beyond goal posts to consider trajectory threatening
        # Factor to shift interception point backward along trajectory
        BACKWARD_SHIFT_FACTOR = 1.5
        MAX_ACCELERATION_BASE = 1500
        MAX_ACCELERATION_DANGER = 6000

        # Goal center position

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
            intersect_y = slope * (4500 - ball_pos[0]) + ball_pos[1]

            # Check if intersection is within goal (with margin) and ball is moving toward goal
            goal_y_min = -1000 / 2
            goal_y_max = 1000 / 2

            if goal_y_min <= intersect_y <= goal_y_max and ball_velocity_x > 0:
                is_moving_toward_goal = True

        # print("is ball moving towards goal?", is_moving_toward_goal, is_team_yellow, robot_id)

        # print(f"\r{is_moving_toward_goal}", end='')

        # Case (b): Ball is moving toward goal - intercept the trajectory
        if 3500 < ball_pos[0] < 4500 and -1000 < ball_pos[1] < 1000:
            distance_to_ball = math.hypot(robot_pos[0]-ball_pos[0], robot_pos[1]-ball_pos[1])
            if distance_to_ball > 110:
                # Move to ball
                vx, vy = robot_movement.go_to(robot_pos, ball_pos, kicker_to_target=True, stop_threshold=5)
                w = robot_movement.face_towards(robot_pos=robot_pos, target=ball_pos, max_angular_acceleration=3)
                kick, dribble = 0, 0
            else:
                # Align to target
                midfielder = wm.get_latest_frame().robots_yellow[2].position
                target_direction = math.atan2(midfielder[1] - ball_pos[1], midfielder[0] - ball_pos[0])
                angle_error = (target_direction - robot_pos[2] + math.pi) % (2 * math.pi) - math.pi

                if abs(math.degrees(angle_error)) > 4:
                    omega = 4.0 * angle_error
                    vx, vy, w, kick, dribble = 0, 0, omega, 0, 1
                else:
                    # Kick the ball out
                    distance_to_target = math.hypot(midfielder[1] - ball_pos[1], midfielder[0] - ball_pos[0])
                    vx, vy, w, dribble = 0, 0, 0, 0
                    kick = distance_to_target / 1000
        else:
            if is_moving_toward_goal:
                # print('A')
                # Ball trajectory line: parametric form
                # Point on line: P = ball_pos + t * ball_velocity
                # Find closest point on this line to goalkeeper's current position

                # Direction vector from ball to robot
                dx = robot_pos[0] - ball_pos[0]
                dy = robot_pos[1] - ball_pos[1]
                # print(f"dx: {dx}    dy: {dy}")
                # print(f"ball_vx: {ball_velocity_x}      ball_vy: {ball_velocity_y}")

                # Project onto ball velocity vector to find closest point parameter t
                dot_product = dx * ball_velocity_x + dy * ball_velocity_y
                velocity_squared = ball_velocity_x**2 + ball_velocity_y**2
                t = dot_product / velocity_squared if velocity_squared > 0 else 0

                # Ensure t is non-negative (don't go behind the ball)
                t = max(0, t)

                # Closest point on trajectory
                closest_x = ball_pos[0] + t * ball_velocity_x
                closest_y = ball_pos[1] + t * ball_velocity_y
                # print(closest_x, closest_y)

                # Calculate distance from robot to closest point
                distance_to_point = math.hypot(robot_pos[0] - closest_x, robot_pos[1] - closest_y)

                # Shift the interception point backward along trajectory proportional to distance
                # This gives the goalkeeper more time to reach the interception point
                shift_distance = distance_to_point * BACKWARD_SHIFT_FACTOR

                # Calculate backward shift in terms of parameter t
                velocity_magnitude = math.hypot(ball_velocity_x, ball_velocity_y)
                if velocity_magnitude > 0:
                    t_shift = shift_distance / velocity_magnitude
                    t = max(0, t - t_shift)

                # Final interception target
                target_x = ball_pos[0] + t * ball_velocity_x
                target_y = ball_pos[1] + t * ball_velocity_y

                max_acceleration = MAX_ACCELERATION_DANGER

            else:
                target_x = 4200
                target_y = goal_center_y + POSITION_FRACTION * (ball_pos[1] - goal_center_y)

                goal_y_min = -1000 / 2
                goal_y_max = 1000 / 2
                target_y = max(goal_y_min, min(goal_y_max, target_y))

                max_acceleration = MAX_ACCELERATION_BASE

            vx, vy = robot_movement.go_to(
                robot_pos=robot_pos, target=(target_x, target_y), max_acceleration=max_acceleration
            )
            w = robot_movement.face_towards(robot_pos=robot_pos, target=ball_pos)
            kick, dribble = 0, 0
        # prev_vx, prev_vy, prev_w = vx, vy, w

        # Send command to robot
        cmd = RobotCommand(robot_id=robot_id, vx=vx, vy=vy, w=w, kick=kick, dribble=dribble)
        sender.send_command(cmd)
