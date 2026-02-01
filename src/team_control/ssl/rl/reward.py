import math
from typing import List, Tuple
from team_control.ssl.vision import Robot, Ball
from team_control.ssl.rl.field import Field


class TeamReward:
    """
    Team-level reward function for MAPPO / CTDE.

    Rewards are computed per team (yellow, blue) and are:
    - shared by all agents on the same team
    - zero-sum where applicable
    """

    def __init__(
        self,
        field: Field,
        collision_dist: float = 200.0,
        possession_dist: float = 150.0,
    ):
        self.field = field
        self.collision_dist = collision_dist
        self.possession_dist = possession_dist

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def compute(
        self,
        robots_yellow: List[Robot],
        robots_blue: List[Robot],
        ball: Ball,
    ) -> Tuple[float, float]:
        """
        Compute rewards for yellow and blue teams.

        Returns:
            reward_yellow, reward_blue
        """
        reward_yellow = 0.0
        reward_blue = 0.0

        # 1. Goal scoring
        ry, rb = self._goal_reward(ball)
        reward_yellow += ry
        reward_blue += rb

        # 2. Goal area constraint
        reward_yellow += self._goal_area_penalty(
            robots_yellow, is_left_goal=True
        )
        reward_blue += self._goal_area_penalty(
            robots_blue, is_left_goal=False
        )

        # 3. Collision penalty
        if self._has_collision(robots_yellow):
            reward_yellow -= 0.02
        if self._has_collision(robots_blue):
            reward_blue -= 0.02

        return reward_yellow, reward_blue

    # --------------------------------------------------
    # Reward components
    # --------------------------------------------------

    def _goal_reward(self, ball: Ball) -> Tuple[float, float]:
        """
        Left goal  -> yellow scores
        Right goal -> blue scores
        """
        x, _ = ball.position
        x_max = self.field.x_max

        if x < -x_max:
            return 1.0, -1.0
        elif x > x_max:
            return -1.0, 1.0
        return 0.0, 0.0

    def _goal_area_penalty(
        self,
        robots: List[Robot],
        is_left_goal: bool,
    ) -> float:
        """
        Exactly one robot must be inside own goal area.
        Otherwise: penalty.
        """
        count = 0
        for r in robots:
            x, y, _ = r.position
            if abs(y) <= self.field.goal_width / 2:
                if is_left_goal and x <= -self.field.x_max + self.field.goal_depth:
                    count += 1
                elif not is_left_goal and x >= self.field.x_max - self.field.goal_depth:
                    count += 1

        if count != 1:
            return -0.05
        return 0.0

    def _has_collision(self, robots: List[Robot]) -> bool:
        """
        Any pair of robots closer than collision distance.
        Penalized once per timestep per team.
        """
        n = len(robots)
        for i in range(n):
            xi, yi, _ = robots[i].position
            for j in range(i + 1, n):
                xj, yj, _ = robots[j].position
                dx = xi - xj
                dy = yi - yj
                if dx * dx + dy * dy < self.collision_dist ** 2:
                    return True
        return False

    def _has_possession(self, robots: List[Robot], ball: Ball) -> bool:
        """
        Team has possession if any robot is within possession distance of the ball.
        """
        bx, by = ball.position
        for r in robots:
            rx, ry, _ = r.position
            dx = rx - bx
            dy = ry - by
            if dx * dx + dy * dy <= self.possession_dist ** 2:
                return True
        return False
