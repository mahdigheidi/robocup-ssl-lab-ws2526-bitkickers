## this is for testing parsing and connectivity
import time

from team_control.game_controller.message import RefereeMessage
from team_control.network.ssl_sockets import GameControl

gc_recv = GameControl()

for i in range(5):
    ref_msg, _ = gc_recv.listen()
    start_time = time.time()
    new_ref_msg = RefereeMessage.from_proto(referee=ref_msg)
    elapsed = (time.time() - start_time) * 1000  # milliseconds
    total_elapsed = (time.time() - new_ref_msg.packet_timestamp / 1_000_000) * 1000
    print(
        f"internal process time :{elapsed:.3f} ms\t\n"
        + f"recv + process time :{total_elapsed:.3f} ms\t\n"
        + f"{(new_ref_msg.command)}\t  {new_ref_msg.stage}\n"
    )
