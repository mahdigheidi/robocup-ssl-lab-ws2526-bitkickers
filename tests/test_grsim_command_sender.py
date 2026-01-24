"""
grSim Command Sender Example
----------------------------
This script demonstrates how to send robot control commands to the grSim simulator.

Usage:
    1. Create a GrSimSender socket to send commands.
    2. Use either:
        - `new_raw_command()` for manually defined movement values.
        - `send_command()` for sending RobotCommand objects.
"""

from team_control.network.sender import GrSimSender
from team_control.network.robot_command import RobotCommand

# Target grSim IP and command port
DESTINATION_IP = "127.0.0.1"
COMMAND_PORT = 20011
IS_YELLOW = True  # Set team color (True = Yellow, False = Blue)

# Initialize grSim sender socket
sender = GrSimSender(ip=DESTINATION_IP, port=COMMAND_PORT, is_yellow=IS_YELLOW)


def example_1(sender: GrSimSender) -> None:
    """
    Example 1: Send a raw movement command to grSim.

    Args:
        sender (GrSimSender): Active grSim sender instance.
    """
    msg = sender.new_raw_command(
        robot_id=0, vx=2.0, vy=1.0, w=1.0, k=1, d=1, use_team_color=True)
    print("Generated raw command:", msg)

    # Send directly
    sender.send(msg)

    # Or encode manually, then send
    encoded = sender.GSC.encode(msg)
    sender.send(encoded)


def example_2(sender: GrSimSender) -> None:
    """
    Example 2: Send a RobotCommand object to grSim.

    Args:
        sender (GrSimSender): Active grSim sender instance.
    """
    command = RobotCommand(robot_id=1, vx=2, vy=3, w=4, kick=1, dribble=0)
    sender.send_command(command)
    print("Sent RobotCommand:", command)


if __name__ == "__main__":
    while True:
        example_1(sender)
        example_2(sender)
