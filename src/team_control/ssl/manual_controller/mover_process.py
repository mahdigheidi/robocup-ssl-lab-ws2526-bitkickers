import traceback

from team_control.network.sender import GrSimSender
from team_control.network.robot_command import RobotCommand
from team_control.robot.movement import RobotMovement
from team_control.network.send_ball_grsim import BallSender


def mover_process(wm, division, sim_ip, cmd_listen_port, robot_id, is_team_yellow) -> None:
    # Initialize grSim command sender
    sender = GrSimSender(ip=sim_ip, port=cmd_listen_port,
                         is_yellow=is_team_yellow)

    version = 0  # Track vision/world model version

    ball_sender = BallSender()

    robot_movement = RobotMovement()

    robot_obj, teammates_obj, opponents_obj, ball_obj = wm.get_robot_context(
        robot_id, is_team_yellow)
    robot_pos = robot_obj.position
    ball_pos = ball_obj.position

    vx, vy, w = 0, 0, 0
    while True:
        # Update world model when new frame is available
        if version < wm.get_version():
            version = wm.get_version()
            try:
                robot_obj, teammates_obj, opponents_obj, ball_obj = wm.get_robot_context(
                    robot_id, is_team_yellow)
                robot_pos = robot_obj.position
                ball_pos = ball_obj.position
            except Exception as e:
                robot_pos = 0, 0, 0
                ball_pos = 0, 0
                traceback.print_exc()

        # Set target position
        target_position = ball_pos

        # Compute velocities to move toward the target
        vx, vy = robot_movement.go_to(
            robot_pos=robot_pos, target=target_position, kicker_to_target=True
        )

        w = robot_movement.face_towards(robot_pos=robot_pos, target=ball_pos)

        # Create and send the command
        cmd = RobotCommand(robot_id=robot_id, vx=vx,
                           vy=vy, w=w, kick=0, dribble=0)
        sender.send_command(cmd)
