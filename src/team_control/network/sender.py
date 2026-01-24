import logging
from team_control.network.base_udp import BaseSocket, SocketType
from team_control.network.grsim_commands import GrSimRobotCommands
from team_control.network.robot_command import RobotCommand


log = logging.getLogger()
log.setLevel(logging.DEBUG)


class Sender(BaseSocket):
    """
    Generic UDP sender socket.

    Attributes:
        destination_ip (str): IP of the remote destination.
        destination_port (int): Port of the remote destination.
    """

    def __init__(self, ip: str = '127.0.0.1', port: int = 0, socket_type=SocketType.SOCK_UDP, binding=False) -> None:
        device_ip = None
        device_port = self._generate_port()
        self.destination_ip = ip
        self.destination_port = port
        super().__init__(ip=device_ip, port=device_port,
                         socket_type=socket_type, binding=binding)

    @property
    def destination_ip(self):
        """Get the destination IP address."""
        return self._destination_ip

    @destination_ip.setter
    def destination_ip(self, value):
        """Set the destination IP address."""
        if not isinstance(value, str):
            raise TypeError("Need IP Address (v4) as string type")
        self._destination_ip = value

    @property
    def destination(self):
        """Return the destination as a tuple (IP, port)."""
        return (self.destination_ip, self.destination_port)

    def _send(self, msg, ip: str = None, port: int = None):
        """
        Send a message via UDP.

        Args:
            msg (str | bytes): Message to send.
            ip (str, optional): Override destination IP.
            port (int, optional): Override destination port.
        """
        addr = tuple(ip, port) if isinstance(
            ip, str) and isinstance(port, int) else self.destination

        if not isinstance(msg, bytes):
            try:
                msg = msg.encode()
            except Exception as e:
                raise ValueError(f"Error encoding message: {e}")

        self.sock.sendto(msg, addr)

    def update_destination(self, destination: str | tuple[str, int]) -> None:
        """
        Update the default destination for future sends.

        Args:
            destination (str | tuple): IP and port, either as a tuple or string "(ip, port)".
        """
        if isinstance(destination, str):
            destination = self.string_to_tuple(destination)
        self.destination_ip = destination[0]
        self.port = destination[1]


class Broadcaster(Sender):
    """
    UDP broadcast sender.
    """

    def __init__(self, broadcasting_port: int = 12342) -> None:
        """
        UDP Broadcast sending channel

        Args:
            broadcasting_port (int, optional): Broadcast channel.
                Defaults to 12342. *This has to be calibrated with recipients

        Params:
            ip (str) : Default broadcast. See python udp_broadcast for more information.
        """
        ip = '<broadcast>'
        socket_type = SocketType.SOCK_BROADCAST_UDP
        super().__init__(ip=ip, port=broadcasting_port,
                         socket_type=socket_type, binding=True)

# ===============================
# GRSIM COMMAND SENDER
# ===============================
class GrSimSender(Sender):
    """
    UDP sender for grSim robot commands.

    Responsible for encoding and sending control packets to the grSim simulator.
    """

    def __init__(self, ip: str = "127.0.0.1", port: int = 20011, is_yellow=True) -> None:
        """
        Initialize grSim sender.

        Args:
            ip (str): Destination IP address of the grSim simulator.
            port (int): Destination port for control commands.
            is_yellow (bool): Whether this sender controls the yellow team.
        """
        self.is_yellow = is_yellow
        self.GSC = GrSimRobotCommands(is_yellow=is_yellow)
        super().__init__(ip=ip, port=port)

    def new_raw_command(self, robot_id, vx=0.0, vy=0.0, w=0.0, k=0, d=0, use_team_color=True):
        """
        Generate a new grSim robot command.

        Returns:
            grSim_Commands_pb2.grSim_Commands: command packet.
        """
        return self.GSC.new_command(robot_id=robot_id, vx=vx, vy=vy, w=w, k=k, d=d, use_team_color=use_team_color)

    def _send(self, msg) -> None:
        """
        Send a command to grSim.

        Args:
            msg (grSim_Commands_pb2.grSim_Commands | bytes): Command to send.
        """
        if not isinstance(msg, bytes):
            try:
                msg = self.GSC.encode(msg)
            except Exception as e:
                raise ValueError(f"Error encoding grSim message: {e}")
        self.sock.sendto(msg, self.destination)

    def send_command(self, robot_command: RobotCommand, use_team_color=True) -> None:
        """
        Convert a RobotCommand object into a grSim command and send it.

        Args:
            robot_command (RobotCommand): The robot command object.
            us (bool): Whether this is for our team (True) or the opponent.
        """
        packet = self.GSC.convert(
            robot_command=robot_command, use_team_color=use_team_color)
        encoded_msg = self.GSC.encode(packet)
        self._send(encoded_msg)
