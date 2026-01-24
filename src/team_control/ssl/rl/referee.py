import time
import random
import math
from multiprocessing import Event

from team_control.network.send_ball_grsim import BallSender

class Referee:
    """
    The Referee is responsible for:
    1. Detecting episode termination conditions:
       - Goal scored
       - Ball out of field
       - Timeout
    2. Halting the game
    3. Resetting robots to formation

    # -------------------------------------------------
    # Geometry helpers
    4. Triggering training
    5. Keeping track of episode time
    """

    def __init__(self, field, wm, halt_event, max_episode_time=30.0, reset_delay=2.5):
        """
        field: Field object
        wm: WorldModel
        max_episode_time: seconds before forced termination
        reset_delay: seconds to wait after halting before reset
        """
        self.field = field
        self.wm = wm
        self.max_episode_time = max_episode_time
        self.reset_delay = reset_delay

        self.episode_start_time = time.time()
        self.halt_event = halt_event
        self.ball_sender = BallSender()
        # time.sleep(2)  # Give some time for initialization

    # -------------------------------------------------
    # Episode lifecycle
    # -------------------------------------------------

    def start_episode(self):
        self.episode_start_time = time.time()
        self.halt_event.clear()
        print("[Referee] Episode started")

    def stop_episode(self):
        self.halt_event.set()

        print("[Referee] Episode ended. Halting game.")

        # Spawning the ball in the center of the field
        theta = random.uniform(0, 2*math.pi)
        r = 500 * math.sqrt(random.random())
        x_ball, y_ball = r * math.cos(theta), r * math.sin(theta)
        self.ball_sender.send_ball_to_grsim(x_ball, y_ball, 0, 0)

    # -------------------------------------------------
    # Termination checks
    # -------------------------------------------------

    def check(self):
        """
        Should be called every control tick.
        """
        # If halted, do nothing until external reset
        if self.halt_event.is_set():
            return

        # Timeout
        if time.time() - self.episode_start_time > self.max_episode_time:
            print("[Referee] Timeout reached")
            self.stop_episode()
            return

        # Ball state
        frame = self.wm.get_latest_frame()
        if frame is None or frame.ball is None:
            return

        ball_pos = frame.ball.position

        # Out of field
        if not self._is_ball_in_field(ball_pos):
            print("[Referee] Ball out of field")
            self.stop_episode()
            return

        # Goal
        if self._is_goal(ball_pos):
            print("[Referee] Goal detected")
            self.stop_episode()
            return
    # -------------------------------------------------
    def _is_ball_in_field(self, ball_pos):
        x, y = ball_pos
        return self.field.x_min <= x <= self.field.x_max and self.field.y_min <= y <= self.field.y_max

    def _is_goal(self, ball_pos):
        x, y = ball_pos
        goal_y_min = -self.field.goal_width / 2
        goal_y_max = self.field.goal_width / 2

        # Left goal
        if x <= self.field.x_min and goal_y_min <= y <= goal_y_max:
            return True

        # Right goal
        if x >= self.field.x_max and goal_y_min <= y <= goal_y_max:
            return True

        return False
