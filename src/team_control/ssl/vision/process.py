import logging
from multiprocessing import Queue

from team_control.ssl.vision.frame import Frame
from team_control.ssl.vision.field import GeometryData
from team_control.network.receiver import Vision
from team_control.utils.logger import LogSaver


class VisionProcess():
    GRSIM_CAMERAS = 4
    REAL_LIFE_CAMERAS = 1

    def __init__(self, output_q: Queue, logs, use_grsim: bool, vision_port):
        self.logs: LogSaver = logs
        self.use_grsim = use_grsim
        self.output_q = output_q
        self.recv = Vision(port=vision_port)
        self.field = None
        self.frame = None
        self.frame_number = -1

    @property
    def cameras(self):
        return self.GRSIM_CAMERAS if self.use_grsim is True else self.REAL_LIFE_CAMERAS

    @property
    def has_field(self):
        return self.field is not None

    def run(self) -> bool:
        while True:
            new_vision_data = self.recv.listen()
            if new_vision_data.HasField("detection"):
                new_detection_data = new_vision_data.detection
                if self.frame_number < new_detection_data.frame_number:
                    self.logs.info(f"{new_detection_data.frame_number=}")
                    # generates new frame
                    self.frame = Frame.from_proto(
                        new_detection_data, self.cameras)
                    self.frame_number = self.frame.frame_number
                if self.frame_number == new_detection_data.frame_number:
                    self.logs.info(
                        f"updating last one, {new_detection_data.camera_id}")
                    self.frame.update(new_detection_data)
                    if self.frame.is_completed:
                        self.send(self.frame)
            if new_vision_data.HasField("geometry"):
                geometry = new_vision_data.geometry
                self.field = GeometryData.from_proto(geometry)
                self.send(self.field)

    def send(self, data):
        if not self.output_q.full():
            self.output_q.put(data)
        else:
            self.logs.warning("QUEUE IS FULL")


def vision_worker(output_q: Queue, use_grsim: bool, vision_port):
    logs = LogSaver(log_level=logging.INFO)
    v = VisionProcess(output_q, logs, use_grsim, vision_port)
    v.run()
