import math
import time
import traceback

from team_control.network.sender import GrSimSender
from team_control.network.robot_command import RobotCommand
from team_control.network.send_ball_grsim import BallSender
from team_control.network.ball_spawner import BallSpawner
from team_control.robot.movement import RobotMovement
from team_control.ssl.datasets.data_generator import DataWriter
from team_control.world.model import WorldModel
from team_control.ssl.rl.env import get_my_normalized_state

# --------------------------
# Ball spawner parameters
# --------------------------
BALL_SPEED = 5.0  # m/s
GOAL_PROBABILITY = 0.75  # probability next ball goes toward goal (0..1)


def goalie_process(wm: WorldModel, division_name, sim_ip, cmd_listen_port, robot_id, is_team_yellow) -> None:
    # Initialize grSim command sender
    data_writer = DataWriter(f"{'yellow' if is_team_yellow else 'blue'}_goalie")
    sender = GrSimSender(ip=sim_ip, port=cmd_listen_port, is_yellow=is_team_yellow)
    # writer = DataWriter('goalie')

    # Track world model
    version = 0
    robot_pos = (0, 0, 0)
    prev_robot_pos = (0, 0, 0)
    ball_pos = (0, 0)
    prev_ball_pos = (-1, -1)

    # Initialize BallSpawner
    spawner = BallSpawner(division_name=division_name, ball_speed=BALL_SPEED, goal_probability=GOAL_PROBABILITY)
    # ball_sender = BallSender()

    robot_obj, ball_obj, teammates_obj, opponents_obj = wm.get_robot_context(robot_id, is_team_yellow)
    ball = wm.get_latest_frame().ball
    action = (0, 0, 0, 0, 0)

    robot_movement = RobotMovement()

    # Track number of goals scored
    goal_count = 0
    goal_center_x = spawner.field.x_max if is_team_yellow else spawner.field.x_min
    goal_center_y = 0

    # Initial ball spawn
    # x, y, vx, vy = spawner.spawn_next_ball()
    # ball_sender.send_ball_to_grsim(x, y, vx, vy)

    t = time.time()
    start_time = time.time()
    prev_vx, prev_vy, prev_w = None, None, None

    while time.time() - start_time < 1200:
        # Update world model when new frame is available
        if version < wm.get_version():
            t = time.time()
            version = wm.get_version()
            try:
                if prev_vx:
                    robot_vel = [(robot_pos[i] - prev_robot_pos[i]) * 0.012 for i in range(len(robot_pos))]
                    robot_vel[2] *= 1000
                    ball_vel = [(ball_pos[i] - prev_ball_pos[i]) * 0.012 for i in range(len(ball_pos))]

                if robot_obj is not None:
                    robot_state, ball_state, teammates_state, opponents_state = get_my_normalized_state(robot_obj, ball_obj, teammates_obj, opponents_obj, is_team_yellow)
                    data_writer.write(robot_state, ball_state, teammates_state, opponents_state, action)
                robot_obj, ball_obj, teammates_obj, opponents_obj = wm.get_robot_context(robot_id, is_team_yellow)
                prev_robot_pos = robot_pos
                robot_pos = robot_obj.position if robot_obj is not None else(goal_center_x, 0)
                prev_ball_pos = ball_pos
                ball_pos = ball_obj.position

            except Exception as e:
                prev_ball_pos = (-1, -1)
                prev_robot_pos = (0, 0, 0)
                ball_pos = 0, 0
                robot_pos = spawner.field.x_max if is_team_yellow else spawner.field.x_min, 0, 0
                traceback.print_exc()
                print(e)

        # ---- GOALKEEPER STRATEGY ----

        # Tunable parameters
        # 0 = at goal center, 1 = at ball (for non-threatening balls)
        POSITION_FRACTION = 0.2
        SPEED_THRESHOLD = 0.5  # m/s - minimum speed to consider ball as "moving toward goal"
        GOAL_MARGIN = 100  # mm - margin beyond goal posts to consider trajectory threatening
        # Factor to shift interception point backward along trajectory
        BACKWARD_SHIFT_FACTOR = 1
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
            intersect_y = slope * (goal_center_x - ball_pos[0]) + ball_pos[1]

            # Check if intersection is within goal (with margin) and ball is moving toward goal
            goal_y_min = -spawner.field.goal_width / 2 - GOAL_MARGIN
            goal_y_max = spawner.field.goal_width / 2 + GOAL_MARGIN

            if goal_y_min <= intersect_y <= goal_y_max and ((ball_velocity_x > 0 and is_team_yellow) or (ball_velocity_x < 0 and not is_team_yellow)):
                is_moving_toward_goal = True

        # print("is ball moving towards goal?", is_moving_toward_goal, is_team_yellow, robot_id)

        # print(f"\r{is_moving_toward_goal}", end='')

        # Case (b): Ball is moving toward goal - intercept the trajectory
        if is_moving_toward_goal:
            # print('A')
            # Ball trajectory line: parametric form
            # Point on line: P = ball_pos + t * ball_velocity
            # Find closest point on this line to goalkeeper's current position

            # Vector from ball to robot
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
            distance_to_line = math.hypot(robot_pos[0] - closest_x, robot_pos[1] - closest_y)

            # Shift the interception point backward along trajectory proportional to distance
            # This gives the goalkeeper more time to reach the interception point
            shift_distance = distance_to_line * BACKWARD_SHIFT_FACTOR

            # Calculate backward shift in terms of parameter t
            velocity_magnitude = math.hypot(ball_velocity_x, ball_velocity_y)
            if velocity_magnitude > 0:
                t_shift = shift_distance / velocity_magnitude
                t = max(0, t - t_shift)
                # print("t is: ", t)

            # Final interception target
            target_x = ball_pos[0] + t * ball_velocity_x
            if is_team_yellow:
                target_x = max(3500, target_x)
            else:
                target_x = min(-3500, target_x)
            target_y = ball_pos[1] + t * ball_velocity_y

            max_acceleration = MAX_ACCELERATION_DANGER
            # print(f"Target_x {target_x}, Target_y {target_y}")

        # Case (a): Ball not threatening - position between ball and goal
        else:
            # print("B")
            # Calculate target position as a fraction along the line from goal center to ball
            target_x = goal_center_x + POSITION_FRACTION * (ball_pos[0] - goal_center_x)
            target_y = goal_center_y + POSITION_FRACTION * (ball_pos[1] - goal_center_y)

            # Clamp target_y to within goal posts to ensure goalkeeper stays in reasonable range
            goal_y_min = -spawner.field.goal_width / 2
            goal_y_max = spawner.field.goal_width / 2
            target_y = max(goal_y_min, min(goal_y_max, target_y))

            max_acceleration = MAX_ACCELERATION_BASE

        # Move to target position
        vx, vy = robot_movement.go_to(
            robot_pos=robot_pos, target=(target_x, target_y), max_acceleration=max_acceleration
        )
        w = robot_movement.face_towards(robot_pos=robot_pos, target=ball_pos)
        prev_vx, prev_vy, prev_w = vx, vy, w

        # Send command to robot
        action = (vx, vy, w, 1, 0)
        cmd = RobotCommand(robot_id=robot_id, vx=vx, vy=vy, w=w, kick=1, dribble=0)
        sender.send_command(cmd)

        # ---- CHECK FOR GOAL ----
        # Check if the ball crossed the goal line and is within the goal posts
        # goal_y_min = -spawner.field.goal_width / 2
        # goal_y_max = spawner.field.goal_width / 2
        # if ball_pos[0] >= spawner.field.x_max and goal_y_min <= ball_pos[1] <= goal_y_max:
        #     goal_count += 1
        #     print(f"Total goals so far: {goal_count}")
        #     # Respawn the ball immediately after a goal
        #     x, y, vx, vy = spawner.spawn_next_ball()
        #     ball_sender.send_ball_to_grsim(x, y, vx, vy)
        #     prev_ball_pos = ball_pos
        #     ball_pos = (x, y)

        # # ---- BALL RESPAWNING LOGIC (DO NOT CHANGE) ----
        # elif BallSpawner.is_ball_idle(ball_pos, prev_ball_pos) or not spawner.is_ball_in_field(ball_pos):
        #     x, y, vx, vy = spawner.spawn_next_ball()
        #     ball_sender.send_ball_to_grsim(x, y, vx, vy)
        #     prev_ball_pos = ball_pos
        #     ball_pos = (x, y)
    data_writer.close()
