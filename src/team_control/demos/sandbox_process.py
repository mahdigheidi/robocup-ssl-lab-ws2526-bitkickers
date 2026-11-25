"""
sandbox_process.py

Example sandbox process for sending movement commands to a robot in grSim.

This script demonstrates how to:
    1. Initialize a grSim sender.
    2. Continuously read the world model (wm) for robot and ball positions.
    3. Compute velocity commands toward a target using RobotMovement.
    4. Send RobotCommand packets to grSim.
"""

import socket

from team_control.motion.movement import RobotMovement
from team_control.network.proto2 import grSim_Packet_pb2, grSim_Replacement_pb2
from team_control.network.robot_command import RobotCommand
from team_control.network.ssl_sockets import grSimSender


def set_robot_position(is_yellow, robot_id, x, y, direction=0.0, address="127.0.0.1", port=20011):
    """
    Set a specific robot to a specific position.

    Args:
        is_yellow (bool): True for yellow team, False for blue team.
        robot_id (int): Robot ID (0-11).
        x (float): X position in mm.
        y (float): Y position in mm.
        direction (float): Direction in radians (default 0.0 = facing right).
        address (str): grSim IP address.
        port (int): grSim replacement port (default 20011).
    """
    replacement = grSim_Replacement_pb2.grSim_Replacement()  # type: ignore

    robot = replacement.robots.add()
    robot.id = robot_id
    robot.x = x / 1000.0  # Convert mm to meters
    robot.y = y / 1000.0  # Convert mm to meters
    robot.dir = direction
    robot.yellowteam = is_yellow
    robot.turnon = True

    # Create packet and send
    packet = grSim_Packet_pb2.grSim_Packet()  # type: ignore
    packet.replacement.CopyFrom(replacement)
    data = packet.SerializeToString()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(data, (address, port))
    sock.close()


def sandbox_process(wm) -> None:
    """
    Main loop for controlling a single robot toward a target in a simulated environment.

    Args:
        wm: WorldModel object that provides robot and ball positions via:
            - wm.get_version()
            - wm.get_yellow_robots()
            - wm.get_latest_frame()

    Behavior:
        - Continuously monitors the world model version to detect updates.
        - Reads the position of the specified robot and the ball.
        - Computes velocities (vx, vy) to move the robot toward the ball.
        - Sends the commands to grSim using a UDP sender.

    Notes:
        - The target is currently set as the ball position.
        - Angular velocity (w), kick, and dribble are kept at 0 for simplicity.
        - Can be adapted for multi-robot or advanced control logic.
    """
    # grSim connection parameters
    SIM_IP = "127.0.0.1"
    CMD_LISTEN_PORT = 20011
    IS_YELLOW = True
    ROBOT_ID = 0  # change this if you want another robot to move

    # Initialize grSim command sender
    sender = grSimSender(ip=SIM_IP, port=CMD_LISTEN_PORT, is_yellow=IS_YELLOW)

    version = 0  # Track vision/world model version
    robot_pos = 0, 0, 0
    ball_pos = 0, 0

    robot_movement = RobotMovement()
    set_robot_position(True, 0, 4000, 0)
    while True:
        # Update world model when new frame is available
        if version < wm.get_version():
            version = wm.get_version()
            try:
                robot_obj = wm.get_yellow_robots(isYellow=IS_YELLOW, robot_id=ROBOT_ID)
                robot_pos = robot_obj.position
                ball_pos = wm.get_latest_frame().ball.position
            except Exception:
                robot_pos = 0, 0, 0
                ball_pos = 0, 0

        # Set target position
        target_position = 0, 0  # Center of the field

        # Compute velocities to move toward the target
        vx, vy = robot_movement.go_to(
            robot_pos=robot_pos, target=target_position, kicker_to_target=True
        )

        w = robot_movement.face_towards(robot_pos=robot_pos, target=ball_pos)

        # Create and send the command
        cmd = RobotCommand(robot_id=0, vx=vx, vy=vy, w=w, kick=0, dribble=0)
        sender.send_command(cmd)
