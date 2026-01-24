""" Receiver - Socket
This file includes :
1. Receiver - Basic UDP receiver w/ Recieving Functions (exc. sending)
2. Broadcast - USP Broadcast receiver
3. Multicast - UDP Multicast receiver
"""
import socket
import struct
import time
import logging

from team_control.network.base_udp import BaseSocket, SocketType
from team_control.network.proto2 import ssl_vision_wrapper_pb2, ssl_gc_referee_message_pb2

# logging
log = logging.getLogger()
log.setLevel(logging.INFO)


class Receiver(BaseSocket):
    def __init__(self,
                 socket_type=SocketType.SOCK_UDP,
                 ip=None,
                 port=0,
                 buffer_size=1024,
                 timeout=1):
        super().__init__(socket_type=socket_type, ip=ip,
                         port=port, binding=True)  # UDP must Bind
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.sock.settimeout(timeout)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} - recv @ {self.addr}, available = {self.is_ready}, {self.timeout=}"

    def listen(self, duration: float = 0.) -> tuple[str, tuple[str, int]] | None:
        """
        Listens for incoming UDP messages. Returns the message and sender address.
        """
        assert self.is_ready, "Socket is not initialized. try another port ?"
        active = True
        result = '', 0
        if duration != 0:
            end_time = time.time() + duration
        else:
            end_time = time.time()
        while active:
            try:
                message, addr = self.sock.recvfrom(self.buffer_size)
                self.last_message = self._decode(message)
                self.last_addr = addr
                result = self.last_message, self.last_addr
                active = False
            except socket.timeout:
                print("timeout")
                continue

            if time.time() >= end_time:
                return result

    def _decode(self, data: bytes):
        try:
            decoded_data = data.decode()
        except UnicodeDecodeError:
            raise UnicodeDecodeError(
                f"cannot decode {data} with default 'utf-8'")
        return decoded_data


# Multicast Receiver
class Multicast(Receiver):
    def __init__(
            self,
            port: int,
            group: str,
            decoder: object,
            buffer_size: int = 6000,
            timeout: int = 0.5) -> None:
        """
        Initialising Multicast socket

        Args:
            decoder (object): SSL Protobuf file and descriptor to decode data
            ip (str, optional): ip of Device trying to connect.
                                Defaults to None -> self obtain device ip.
                                if wanted to connect externally, please type in ip string.
            port (int, optional): port of connection. Defaults to 0 -> auto generated.
            group (str, optional): Mulitcast group. Defaults to "224.5.0.0" -> nothing.
            buffer_size (int, optional): buffer_size for listening. Defaults to 6000.

        Raises:
            Exception: Needed decoder
        """
        super().__init__(ip="", port=port, socket_type=SocketType.SOCK_MULTICAST_UDP,
                         buffer_size=buffer_size, timeout=timeout)

        self.group: str = group
        self.decoder: object = decoder
        self._add_group()

    def _add_group(self):
        """adds group to multicast socket"""
        self.is_ready = False
        mreq = struct.pack("=4sl", socket.inet_aton(
            self.group), socket.INADDR_ANY)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        self.is_ready = True

    def listen(self) -> str | None:  # modified to have decoded data to be boolean
        return super().listen()

    def _decode(self, data: bytes) -> bool:
        """
        Reading and decoding the data (overrides super)

        Args:
            data (bytes): data received from listen ()

        Returns:
            str: decoded data
        """
        # Decode with Protobuf
        decoded_data: str = self.decoder.FromString(data)
        return decoded_data


# ===============================
# SSL-VISION RECEIVERS
# ===============================
class Vision(Multicast):
    """
    SSL-Vision multicast receiver.

    Receives raw SSL-Vision packets via UDP multicast.
    These packets contain data about detected balls and robots from the vision system.

    Args:
        port (int): Port for the SSL-Vision multicast. Defaults to 10006.
    """

    def __init__(self, port: int) -> None:
        decoder: object = ssl_vision_wrapper_pb2.SSL_WrapperPacket()  # pylint: disable=no-member
        group: str = "224.5.23.2"
        buffer_size: int = 6000
        super().__init__(port=port, group=group, decoder=decoder, buffer_size=buffer_size)

    def listen(self) -> ssl_vision_wrapper_pb2.SSL_WrapperPacket:  # pylint: disable=no-member
        """
        Listen for incoming SSL-Vision packets.

        Returns:
            SSL_WrapperPacket: decoded vision data.
        """
        vision_data, addr = super().listen()
        return vision_data


# ===============================
# SSL-GameController RECEIVERS
# ===============================
class GameControl(Multicast):
    """
    SSL-GameController multicast receiver.

    Receives referee messages containing play state, goals, team info, etc.
    """

    def __init__(self, port: int) -> None:
        group: str = '224.5.23.1'
        decoder = ssl_gc_referee_message_pb2.Referee()  # pylint: disable=no-member
        buffer_size: int = 6000
        timeout: float = 5.0
        super().__init__(port=port, group=group,
                         decoder=decoder, buffer_size=buffer_size, timeout=timeout)

    def listen(self) -> ssl_gc_referee_message_pb2.Referee:  # pylint: disable=no-member
        """
        Listen for referee messages.

        Returns:
            Referee: decoded referee message protobuf.
        """
        data, addr = super().listen()
        return data
