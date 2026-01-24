"""
world_model.py

Central World Model (wm) for the SSL robots.

This module provides an interface for robot and game state information.
It maintains vision frames, field geometry, and game controller updates,
allowing other processes to query or act on the latest world state.

Features:
    - Multiprocessing safe with Manager values.
    - History of frames stored for temporal queries.
    - Keeps track of active robots, team assignment, and ball location.
    - Provides helpers for accessing our team and opponent robots.

Author: Emma
"""

import logging
from multiprocessing import Manager

from team_control.ssl.vision.frame_list import FrameList
from team_control.ssl.vision.field import GeometryData, FieldSize
from team_control.ssl.vision.frame import Frame

import copy


log = logging.getLogger()
log.setLevel(logging.INFO)


class WorldModel:
    """
    WorldModel (wm) maintains the global state of the SSL environment.

    Attributes:
        update_interval (int): Number of frames before incrementing version.
        use_sim (bool): Whether the world model is running in simulation mode.
        frame_list (FrameList[Frame]): Historical list of frames.
        geometry (GeometryData): Field and model geometry.
        field (FieldSize): Field dimensions.
        _version (Manager.Value): Version counter for multiprocessing.
        robot_active (int): Number of active robots.
        game_state (GameState): Current game state.
        _us_yellow (bool): Whether our team is yellow.
        _us_positive (bool): Whether our coordinate system is positive.
        blf_location (tuple): Last recorded ball-left-field location.
    """

    def __init__(
        self,
        update_interval: int = 5,
        history: int = 60,
        use_sim: bool = True,
        us_yellow=True,
        us_positive=True,
    ):
        mgr = Manager()
        self._us_yellow = us_yellow
        self._us_positive = us_positive
        self.count = 0
        self.update_interval: int = update_interval
        self.use_sim: bool = use_sim
        self.frame_list: FrameList = FrameList(history=history)
        self.geometry: GeometryData = None
        self.field: FieldSize = None
        self._version = mgr.Value('i', 0)  # int counter
        self.robot_active = 6  # robots active
        self.game_state = None
        self.blf_location = None
        # self.logger = logSaver()

    # ------ Frame management ------
    def add_new_frame(self, frame: Frame):
        """Add a new vision frame and update version periodically."""
        self.count += 1
        if self.count >= self.update_interval:
            self._version.value += 1
            self.count = 0
        self.frame_list.append(frame)

    def get_latest_frame(self) -> Frame:
        """Return the most recent vision frame."""
        return self.frame_list.latest

    def get_last_n_frames(self, n: int):
        """Return the last n frames for temporal analysis."""
        return self.frame_list.get_last_n_frames(n)

    # ------ Geometry ------
    def update_geometry(self, geometry: GeometryData):
        self.geometry = geometry
        self.field = geometry.field
        self.ball_model = geometry.models

    def update_robots_active(self, new_active: int):
        """Update the number of active robots on the field."""
        self.robot_active = new_active

    def update_state(self, new_state):
        """Update current game state."""
        self._state = new_state

    def update_team(self, us_yellow: bool, us_positive: bool):
        """Update team assignment and coordinate polarity."""
        self._us_yellow = us_yellow
        self._us_positive = us_positive

    def update_ball_left_field_location(self, location):
        """Store the last ball-left-field location."""
        self.blf_location = location

    def get_ball_left_field_location(self):
        return self.blf_location

    def get_current_state(self):
        return self._state

    def us_yellow(self):
        return self._us_yellow

    def us_positive(self):
        return self._us_positive

    def get_our_robots(self, us=True, robot_id=None):
        """Return our team robots based on `us_yellow` setting."""
        frame = self.frame_list.latest
        # use is_yellow value as us_yellow if us== True, otherwise, the opposite i.e. not(us_yellow)
        is_yellow = self._us_yellow if us else not self._us_yellow
        # get list of robots base on team color
        robots = frame.robots_yellow if is_yellow else frame.robots_blue
        # returns a robot if a valid id is given, otherwise, returns list of robot
        return robots[robot_id] if isinstance(robot_id, int) else robots

    def get_active_robots(self):
        return self.robot_active

    def get_version(self):
        """Return the current version of the world model."""
        return self._version.value

    def get_robot_context(self, robot_id, is_yellow):
        team_obj = self.get_our_robots(us=is_yellow)
        robot_obj = team_obj.get_robot(robot_id)
        ball_obj = self.get_latest_frame().ball
        teammates_obj = copy.deepcopy(team_obj)
        teammates_obj = teammates_obj.remove(robot_id)
        opponents_obj = self.get_our_robots(not is_yellow)
        return robot_obj, ball_obj, teammates_obj, opponents_obj
