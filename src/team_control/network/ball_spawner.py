"""
Ball spawning logic for grSim sandbox process.
"""

import random

from team_control.ssl.rl.field import Field


class BallSpawner:
    """
    BallSpawner handles spawning balls for a grSim sandbox simulation.

    It can spawn balls either towards the goal or at random locations within
    the field. Velocities, field size, and goal parameters depend on the chosen
    division ('A' or 'B').

    Parameters
    ----------
    division_name : str
        Field division, either 'A' or 'B' or 'S'. Determines field size and goal width.
    ball_speed : float
        Base velocity magnitude for spawned balls in m/s.
    goal_probability : float
        Probability [0,1] that the next spawned ball goes towards the goal
        instead of a random target.

    Attributes
    ----------
    field_x_min, field_x_max : float
        Minimum and maximum X coordinates of the field (in mm).
    field_y_min, field_y_max : float
        Minimum and maximum Y coordinates of the field (in mm).
    goal_line_x : float
        X position of the goal line (in mm).
    goal_width : float
        Width of the goal (in mm).
    """

    def __init__(self, division_name, ball_speed=5.0, goal_probability=0.5, seed=None):
        self.field = Field(division_name)
        self.ball_speed = ball_speed
        self.goal_probability = goal_probability
        if seed is None:
            seed = random.randrange(0, 10000)

        self.random_gen = random.Random(seed)

    def spawn_next_idle_ball(self, robot_pos):
        """Return (x, y) for the next ball."""
        ball_pos_max_x = 0.9 * self.field.x_max / 1000
        ball_pos_min_x = 0.9 * self.field.x_min / 1000
        ball_pos_max_y = 0.9 * self.field.y_max / 1000
        ball_pos_min_y = 0.9 * self.field.y_min / 1000
        ball_x = self.random_gen.uniform(ball_pos_min_x, ball_pos_max_x)
        ball_y = self.random_gen.uniform(ball_pos_min_y, ball_pos_max_y)
        # Don't spawn the ball at the position of the agent
        while (robot_pos[0] - 0.09 < ball_x < robot_pos[0] + 0.09 and
               robot_pos[1] - 0.09 < ball_y < robot_pos[1] + 0.09):
            ball_x = self.random_gen.uniform(ball_pos_min_x, ball_pos_max_x)
            ball_y = self.random_gen.uniform(ball_pos_max_y, ball_pos_min_y)

        return ball_x, ball_y

    def spawn_next_ball(self):
        """Return (x, y, vx, vy) for the next ball."""
        if self.random_gen.random() < self.goal_probability:
            return self.shoot_at_goal()
        return self.shoot_at_random_target()

    def shoot_at_goal(self):
        """Shoot a ball toward a random point within the goal."""
        ball_x = self.random_gen.randint(0, 3000)
        ball_y = self.random_gen.randint(-2000, 2000)

        target_x = self.field.x_max
        target_y = self.random_gen.uniform(-self.field.goal_width /
                                           2, self.field.goal_width / 2)

        ratio = (target_y - ball_y) / (target_x - ball_x)
        vx = self.ball_speed
        vy = self.ball_speed * ratio
        return ball_x, ball_y, vx, vy

    def shoot_at_random_target(self):
        """Shoot a ball toward a random point in the field."""
        ball_x = self.random_gen.uniform(0, 1)
        ball_y = self.random_gen.uniform(-1, 1)

        ratio = self.random_gen.uniform(-1, 1)
        vx = self.ball_speed
        vy = self.ball_speed * ratio
        return ball_x, ball_y, vx, vy

    def is_ball_in_field(self, pos):
        """Check if a ball is within the field boundaries."""
        x, y = pos
        return self.field.x_min <= x <= self.field.x_max and self.field.y_min <= y <= self.field.y_max

    @staticmethod
    def is_ball_idle(current, previous):
        """Return True if the ball is nearly stationary."""
        ball_x, ball_y = current
        prev_x, prev_y = previous
        if ball_x == 0:
            return False
        return abs(ball_x - prev_x) + abs(ball_y - prev_y) < 0.2
