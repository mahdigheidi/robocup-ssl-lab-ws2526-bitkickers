import logging
from multiprocessing import Queue

from team_control.ssl.vision.field import GeometryData
from team_control.ssl.vision.frame import Frame
from team_control.world.model import WorldModel
from team_control.utils.logger import LogSaver


def wm_runner(wm: WorldModel, vision_q: Queue, gc_q: Queue):
    logs = LogSaver(log_level=logging.INFO)
    while True:
        try:
            # Add vision data from multicast receiver
            if not vision_q.empty():
                item = vision_q.get_nowait()
                if isinstance(item, Frame):
                    wm.add_new_frame(item)
                elif isinstance(item, GeometryData):
                    wm.update_geometry(item)
                else:
                    logs.warning(f"Recv unexpected item : {item}")

        except Exception as e:
            logs.error(f"Unexpected Exception caught, {e}")
