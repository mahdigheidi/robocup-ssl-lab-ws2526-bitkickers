import socket
from team_control.network.proto2 import grSim_Replacement_pb2
from team_control.network.proto2 import grSim_Packet_pb2


class BallSender:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.address = "127.0.0.1"
        self.port = 20011

    def __del__(self):
        self.sock.close()

    def send_ball_to_grsim(self, x, y, vx, vy):
        """
        Spawn or move the ball in grSim to position (x, y)
        with velocity (vx, vy).

        Args:
            x (float): X position in mm.
            y (float): Y position in mm.
            vx (float): Velocity in X direction in m/s.
            vy (float): Velocity in Y direction in m/s.
        """

        # Create the BallReplacement message
        ball = grSim_Replacement_pb2.grSim_BallReplacement()  # pylint: disable=no-member
        ball.x = x / 1000
        ball.y = y / 1000
        ball.vx = vx
        ball.vy = vy

        # Create the Replacement message
        replacement = grSim_Replacement_pb2.grSim_Replacement()  # pylint: disable=no-member
        replacement.ball.CopyFrom(ball)

        # Create the Packet message
        packet = grSim_Packet_pb2.grSim_Packet()  # pylint: disable=no-member
        packet.replacement.CopyFrom(replacement)

        # Serialize to bytes
        data = packet.SerializeToString()

        # Send via UDP
        self.sock.sendto(data, (self.address, self.port))
