import time
import math
from TeamControl.network.ssl_sockets import grSimSender
from TeamControl.network.robot_command import RobotCommand
from TeamControl.robot.Movement import RobotMovement
from TeamControl.robot.spawn_robot import send_robot_to_grsim


def interceptor_process(wm):
    SIM_IP = "127.0.0.1"
    CMD_LISTEN_PORT = 20011

    sender = grSimSender(ip=SIM_IP, port=CMD_LISTEN_PORT, is_yellow=False)
    robot_movement = RobotMovement()

    version = 0

    robot_pos = (0, 0, 0)
    yellow_robot0_pos = (0, 0, 0)
    yellow_robot1_pos = (0, 0, 0)
    ball_pos = (0, 0)

    send_robot_to_grsim(team_yellow=False, robot_id=0, x=-1500, y=0, orientation=0)
    send_robot_to_grsim(team_yellow=False, robot_id=1, x=-5000, y=-5000, orientation=0)

    while True:
        if version < wm.get_version():
            version = wm.get_version()
            try:
                robot_pos = wm.get_yellow_robots(isYellow=False, robot_id=0).position
                yellow_robot0_pos = wm.get_yellow_robots(isYellow=True, robot_id=0).position
                yellow_robot1_pos = wm.get_yellow_robots(isYellow=True, robot_id=1).position
                ball_pos = wm.get_latest_frame().ball.position
            except Exception:
                pass

        # TODO: Implement actual logic
        vx, vy = robot_movement.go_to(robot_pos, (0, 0))
        w = robot_movement.face_towards(robot_pos, ball_pos)

        cmd = RobotCommand(robot_id=0, vx=vx, vy=vy, w=w, kick=1, dribble=0)
        sender.send_command(cmd)
