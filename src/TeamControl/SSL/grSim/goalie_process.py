"""
sandbox_process_ball_spawning.py

Example sandbox process for sending movement commands to a robot in grSim.

This script demonstrates how to:
    1. Initialize a grSim sender.
    2. Continuously read the world model (wm) for robot and ball positions.
    3. Spawn balls and respawn them automatically, tracking goals.
"""
from TeamControl.network.ssl_sockets import grSimSender
from TeamControl.network.robot_command import RobotCommand
from TeamControl.ball.send_ball_grsim import send_ball_to_grsim
from TeamControl.ball.ball_spawner import BallSpawner
from TeamControl.robot.Movement import RobotMovement

# --------------------------
# Ball spawner parameters
# --------------------------
DIVISION = 'B'          # 'A' or 'B'
BALL_SPEED = 5.0        # m/s
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
        division=DIVISION,
        ball_speed=BALL_SPEED,
        goal_probability=GOAL_PROBABILITY
    )

    # Track number of goals scored
    goal_count = 0

    # Initial ball spawn
    x, y, vx, vy = spawner.spawn_next_ball()
    send_ball_to_grsim(x, y, vx, vy)

    while True:
        # Update world model when new frame is available
        if version < wm.get_version():
            version = wm.get_version()
            try:
                robot_obj = wm.get_yellow_robots(isYellow=IS_YELLOW,
                                                 robot_id=ROBOT_ID)
                robot_pos = robot_obj.position
                prev_ball_pos = ball_pos
                ball_pos = wm.get_latest_frame().ball.position

            except Exception:
                robot_pos = 0, 0, 0
                prev_ball_pos = (-1, -1)
                ball_pos = 0, 0

        # ---- GOALKEEPER STRATEGY ----
        # ...

        # Firstly we will calculate the point that the ball is heading using its current position
        # For That we will have to calculate the line equation of the ball trajectory
        # We also have the line equation of the goal line which is x = goal_line_x
        # We can find the intersection point of these two lines to find where the ball will cross
        # if the intersection point of the goal line and the ball trajectory is within the goal posts
        # then the goalie will move to that point, otherwise it will stay in the center of the goal
        target_x, target_y = spawner.goal_line_x, 0  # Default to center of goal

        # We also would like to define an exit strategy for the goalie
        # we define a value e, which is a small distance before the goal line to position the goalie
        exit_max, risk_prob = 500, 0.5
        e = exit_max * risk_prob  # mm before the goal line
        # now if the ball is from a distance less than e from the goal line, we will set target_x to be
        # goal_line_x - e
        # if the ball is further than e from the goal line, and is less than a quarter of the field length away,
        # we will set target_x to be e
        # else, if the ball is further than a quarter of the field length away but still in our half.
        # we use this formula to set target_x = -4 * (e / field_length) * ball_x + 2*e
        field_length = spawner.field_x_max - spawner.field_x_min
        ball_distance_to_goal = spawner.goal_line_x - ball_pos[0]
        if ball_distance_to_goal <= e:
            target_x = spawner.goal_line_x - e
        elif ball_distance_to_goal <= field_length / 4:
            target_x = e
        elif ball_distance_to_goal <= field_length / 2:
            target_x = -4 * (e / field_length) * ball_pos[0] + 2 * e


        if ball_pos[0] != prev_ball_pos[0]:  # Avoid division by zero
            # we will proceed using the equation of a line y = ax + b
            a = (ball_pos[1] - prev_ball_pos[1]) / (ball_pos[0] - prev_ball_pos[0])  # slope
            target_y = a * target_x + (ball_pos[1] - a * ball_pos[0])

            # Clamp target_y to within goal posts
            goal_y_min = -spawner.goal_width / 2
            goal_y_max = spawner.goal_width / 2
            if target_y < goal_y_min:
                target_y = goal_y_min
            elif target_y > goal_y_max:
                target_y = goal_y_max
        else:
            target_y = 0  # Ball is moving vertically, stay in center
        
        # now we have target_x and target_y for the goalie to move to
        vx, vy, w = RobotMovement.goToPoint(robot_pos=robot_pos,
                                        target=(target_x, target_y),
                                        slow_radius=500.0)
        # now we send the command to the robot
        cmd = RobotCommand(robot_id=ROBOT_ID, vx=vx, vy=vy, w=w, kick=0, dribble=0)
        sender.send_command(cmd)

        # ---- CHECK FOR GOAL ----
        # Check if the ball crossed the goal line and is within the goal posts
        goal_y_min = -spawner.goal_width / 2
        goal_y_max = spawner.goal_width / 2
        if ball_pos[0] >= spawner.goal_line_x and goal_y_min <= ball_pos[1] <= goal_y_max:
            goal_count += 1
            print(f"Total goals so far: {goal_count}")
            # Respawn the ball immediately after a goal
            x, y, vx, vy = spawner.spawn_next_ball()
            send_ball_to_grsim(x, y, vx, vy)
            prev_ball_pos = ball_pos
            ball_pos = (x, y)


        # ---- BALL RESPAWNING LOGIC (DO NOT CHANGE) ----
        elif BallSpawner.is_ball_idle(ball_pos, prev_ball_pos) or not spawner.is_ball_in_field(ball_pos):
            x, y, vx, vy = spawner.spawn_next_ball()
            send_ball_to_grsim(x, y, vx, vy)
            prev_ball_pos = ball_pos
            ball_pos = (x, y)
