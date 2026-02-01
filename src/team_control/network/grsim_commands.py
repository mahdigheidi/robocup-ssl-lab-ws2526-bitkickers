"""
GrSimRobotCommands.py

Wrapper for generating grSim-compatible robot commands.

This module converts internal RobotCommand objects or raw parameters into
grSim protobuf messages suitable for sending to the grSim simulator.

Features:
- Generate grSim robot commands with velocities, kick, and dribble.
- Convert internal RobotCommand objects to grSim commands.
- Encode commands into serialized bytes for network transmission.
- Support for dynamically changing team color.

Author: Emma
"""

import time
from team_control.network.robot_command import RobotCommand
from team_control.network.proto2 import grSim_Commands_pb2, grSim_Packet_pb2


class GrSimRobotCommands:
    """
    Generates grSim robot commands for one team.

    Attributes:
        is_yellow (bool): True if commands are for yellow team, False for blue.
        kick_speed_x (float): Default kick speed in x direction.
        kick_speed_z (float): Default kick speed in z direction (chip kick).
        wheel_speed (bool): Whether to use individual wheel speed mode.
    """

    def __init__(self, is_yellow: bool):
        """
        Initialize the GrSimRobotCommands (GSC) object.

        Args:
            is_yellow (bool): Team color. True for yellow, False for blue.
        """
        self.is_yellow: bool = is_yellow
        self.kick_speed_x: float = 10.0
        self.kick_speed_z: float = 0.0  # we have no chip kicker at the moment
        self.wheel_speed: bool = False

    def update_is_yellow(self, new_is_yellow: bool) -> None:
        """
        Update the team color dynamically.

        Args:
            new_is_yellow (bool): New team color.
        """
        self.is_yellow = new_is_yellow

    def _robot_command_wrapper(self, robot_id: int, vx: float, vy: float, w: float, k: float, d: bool) -> object:
        """
        Wrap raw command parameters into a grSim_Robot_Command protobuf.

        Args:
            robot_id (int): Robot ID.
            vx (float): Velocity in X direction.
            vy (float): Velocity in Y direction.
            w (float): Angular velocity.
            k (float): Kicking velocity (0 - 10)
            d (bool): Dribble flag.

        Returns:
            grSim_Robot_Command: GrSim protobuf command.
        """
        robot_id = int(robot_id)
        vx, vy, w = float(vx), float(vy), float(w)
        k, d = float(k), bool(d)

        grsim_robot_command = grSim_Commands_pb2.grSim_Robot_Command(  # pylint: disable=no-member
            id=robot_id,
            kickspeedx=k,
            kickspeedz=0.0,
            veltangent=vx,
            velnormal=vy,
            velangular=w,
            spinner=d,
            wheelsspeed=self.wheel_speed
        )
        return grsim_robot_command

    @staticmethod
    def _commands_wrapper(is_yellow, grsim_robot_command) -> object:
        """
        Wrap one or more grSim_Robot_Command objects into a grSim_Commands protobuf.

        Args:
            is_yellow (bool): Team color.
            grSim_robot_command: grSim_Robot_Command object.

        Returns:
            grSim_Commands: GrSim protobuf message.
        """
        grsim_commands = grSim_Commands_pb2.grSim_Commands(  # pylint: disable=no-member
            timestamp=time.time(),
            isteamyellow=is_yellow,
            robot_commands=[grsim_robot_command]
        )
        return grsim_commands

    def new_command(self, robot_id, vx, vy, w, k, d, use_team_color=True):
        """
        Generate a new grSim command from raw parameters.

        Args:
            robot_id (int): Robot ID.
            vx (float): Velocity in X.
            vy (float): Velocity in Y.
            w (float): Angular velocity.
            k (bool): Kick flag.
            d (bool): Dribble flag.
            use_team_color (bool): Use this team's preset color if True, else opposite.

        Returns:
            grSim_Commands: GrSim protobuf message.
        """
        is_yellow = self.is_yellow if use_team_color is True else not self.is_yellow
        grsim_robot_command = self._robot_command_wrapper(
            robot_id, vx, vy, w, k, d)
        grsim_commands = self._commands_wrapper(is_yellow, grsim_robot_command)
        return grsim_commands

    def convert(self, robot_command: RobotCommand, use_team_color=True) -> object:
        """
        Convert an internal RobotCommand object to grSim command.

        Args:
            robot_command (RobotCommand): Internal robot command object.
            use_team_color (bool): Use this team's preset color if True, else opposite.

        Returns:
            grSim_Commands: GrSim protobuf message.
        """
        grsim_command = self.new_command(
            robot_id=robot_command.robot_id,
            vx=robot_command.vx,
            vy=robot_command.vy,
            w=robot_command.w,
            k=robot_command.kick,
            d=robot_command.dribble,
            use_team_color=use_team_color)
        return grsim_command

    def encode(self, grsim_commands) -> bytes:
        """
        Encode grSim command into a serialized packet for network transmission.

        Args:
            grsim_commands: GrSim_Commands protobuf message.

        Returns:
            bytes: Serialized byte string.
        """
        grsim_packet = grSim_Packet_pb2.grSim_Packet(  # pylint: disable=no-member
            commands=grsim_commands)
        byte_packet = grsim_packet.SerializeToString()
        return byte_packet

    def decode(self, grsim_commands) -> object:
        """
        Decode serialized grSim packet back into protobuf object.

        Args:
            byte_data (bytes): Serialized grSim packet.

        Returns:
            grSim_Packet: Decoded protobuf object.
        """
        decoder = grSim_Packet_pb2.grSim_Packet()  # pylint: disable=no-member
        grsim_commands: str = decoder.FromString(grsim_commands)
        return grsim_commands

# --------------------------
# Example Usage
# --------------------------


def example_1():
    """
    Example 1: Generate a new raw grSim command using new_command.
    """
    is_yellow = True
    GSC = GrSimRobotCommands(is_yellow=is_yellow)  # remember ! !
    # Generate a command for robot 1 with velocities, kick, and dribble
    msg1 = GSC.new_command(robot_id=1, vx=2.0, vy=1.0,
                           w=1, k=1, d=1, use_team_color=True)
    print("Raw grSim command:", msg1)

    # Encode the command for sending over the network
    encoded = GSC.encode(msg1)
    print("Encoded packet:", encoded)


def example_2():
    """
    Example 2: Convert an internal RobotCommand object to grSim command.
    """
    is_yellow = True
    GSC = GrSimRobotCommands(is_yellow=is_yellow)

    # Create internal robot command
    r1 = RobotCommand(robot_id=1, vx=2, vy=3, w=4, kick=True, dribble=False)

    # Convert to grSim command
    msg2 = GSC.convert(r1)

    # Encode for transmission
    encoded = GSC.encode(msg2)
    print("Converted grSim command:", msg2)
    print("Encoded packet:", encoded)


def example_3():
    """
    Example 3: Generate a grSim command for the opposite team color.
    """
    is_yellow = True
    GSC = GrSimRobotCommands(is_yellow=is_yellow)

    # Generate a command for the opposite team
    msg3 = GSC.new_command(robot_id=1, vx=2.0, vy=1.0,
                           w=1, k=True, d=True, use_team_color=False)
    print("Opposite team grSim command:", msg3)

    # Encode for sending
    encoded = GSC.encode(msg3)
    print("Encoded packet:", encoded)


def example_4():
    """
    Example 4: Dynamically update team color and generate a command.
    """
    is_yellow = True
    GSC = GrSimRobotCommands(is_yellow=is_yellow)

    print("isYellow before update:", GSC.is_yellow)
    GSC.update_is_yellow(not is_yellow)
    print("is Yellow after update:", GSC.is_yellow)

    msg4 = GSC.new_command(robot_id=1, vx=2.0, vy=1.0,
                           w=1, k=True, d=True, use_team_color=True)
    print("Command with updated team color:", msg4)
    print("Packet team is yellow:", msg4.isteamyellow)


if __name__ == "__main__":
    example_1()
    example_2()
    example_3()
    example_4()
