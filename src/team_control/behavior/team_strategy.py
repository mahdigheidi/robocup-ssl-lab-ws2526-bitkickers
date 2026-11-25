"""
DECISION TREE:
 ├─ Ball in my section?
 │   ├─ I own ball → Shoot or Pass
 │   ├─ Opponent owns ball → Pressure
 │   └─ Free ball → Go to ball
 └─ Ball not in my section
     └─ Move to safest grid point
"""

from dataclasses import dataclass

import numpy as np

from team_control.motion.movement import RobotMovement
from team_control.network.robot_command import RobotCommand


# ============================================================
# Data Containers
# ============================================================
@dataclass
class FieldSection:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def contains(self, pos: np.ndarray) -> bool:
        return self.xmin <= pos[0] <= self.xmax and self.ymin <= pos[1] <= self.ymax


@dataclass
class StrategyConfig:
    possession_distance: float
    possession_angle_threshold: float  # radians
    pressure_distance: float
    max_kick_power: float = 10.0


# ============================================================
# Base Strategy
# ============================================================


class BaseStrategy:
    def __init__(
        self,
        robot_id: int,
        is_yellow: bool,
        section: FieldSection,
        field_limits: FieldSection,
        goal_points: list[np.ndarray],
        config: StrategyConfig,
    ):
        self.robot_id = robot_id
        self.is_yellow = is_yellow
        self.section = section
        self.field_limits = field_limits
        self.goal_points = goal_points
        self.cfg = config

        self.movement = RobotMovement()

    # ========================================================
    # Main entry point
    # ========================================================

    def decide(self, world_state, sender):
        robot = world_state.get_robot_state(self.robot_id, self.is_yellow)
        ball = world_state.ball

        if robot is None or ball is None:
            return

        robot_pos = robot.position
        robot_theta = robot.orientation
        ball_pos = ball.position

        # ---- Decision Tree Root ----
        if self.section.contains(ball_pos):
            cmd = self._ball_in_section(robot_pos, robot_theta, ball_pos, world_state)
        else:
            cmd = self._ball_not_in_section(robot_pos, world_state)

        sender.send_command(cmd)

    # ========================================================
    # Ball in section
    # ========================================================

    def _ball_in_section(self, robot_pos, robot_theta, ball_pos, world_state):
        if self._agent_owns_ball(robot_pos, robot_theta, ball_pos):
            return self._agent_owns_ball_decision(robot_pos, robot_theta, world_state)

        opponent = self._opponent_owns_ball(ball_pos, world_state)
        if opponent is not None:
            return self._pressure_opponent(robot_pos, opponent.position)

        return self._go_to_ball(robot_pos, ball_pos)

    # ========================================================
    # Ball not in section
    # ========================================================

    def _ball_not_in_section(self, robot_pos, world_state):
        target = self._select_safest_point(world_state)
        vx, vy = self.movement.go_to(robot_pos=robot_pos, target=target)
        w = 0

        return RobotCommand(
            robot_id=self.robot_id,
            vx=vx,
            vy=vy,
            w=w,
            kick=0,
            dribble=0,
        )

    # ========================================================
    # Ownership logic
    # ========================================================

    def _agent_owns_ball(self, robot_pos, robot_theta, ball_pos) -> bool:
        dist = np.linalg.norm(ball_pos - robot_pos[:2])
        if dist > self.cfg.possession_distance:
            return False

        # Facing check
        to_ball = ball_pos - robot_pos[:2]
        angle_to_ball = np.arctan2(to_ball[1], to_ball[0])
        angle_diff = np.arctan2(
            np.sin(angle_to_ball - robot_theta),
            np.cos(angle_to_ball - robot_theta),
        )

        return abs(angle_diff) < self.cfg.possession_angle_threshold

    def _opponent_owns_ball(self, ball_pos, world_state):
        for opp in world_state.get_opponents(is_yellow=self.is_yellow):
            if np.linalg.norm(ball_pos - opp.position[:2]) < self.cfg.possession_distance:
                return opp
        return None

    # ========================================================
    # Agent owns ball
    # ========================================================

    def _agent_owns_ball_decision(self, robot_pos, robot_theta, world_state):
        has_shot, goal_point = self._has_clean_shot(robot_pos, world_state)

        if has_shot:
            w = self.movement.face_towards(robot_pos, goal_point)
            return RobotCommand(
                robot_id=self.robot_id,
                vx=0,
                vy=0,
                w=w,
                kick=self.cfg.max_kick_power,
                dribble=0,
            )

        pass_point = self._select_best_pass_point(robot_pos, world_state)
        w = self.movement.face_towards(robot_pos, pass_point)
        kick_power = self._pass_kick_power(robot_pos, pass_point)

        return RobotCommand(
            robot_id=self.robot_id,
            vx=0,
            vy=0,
            w=w,
            kick=kick_power,
            dribble=0,
        )

    # ========================================================
    # Shooting
    # ========================================================
    def _has_clean_shot(self, robot_pos, world_state):
        obstacles = world_state.get_opponents(is_yellow=self.is_yellow) + [
            world_state.get_opponent_goalie(is_yellow=self.is_yellow)
        ]

        for p in self.goal_points:
            if self._trajectory_clear(robot_pos[:2], p, obstacles):
                return True, p
        return False, None

    # ========================================================
    # Passing
    # ========================================================
    def _select_best_pass_point(self, robot_pos, world_state):
        points = self._generate_grid_points()
        scores = [self._score_point(robot_pos, p, world_state) for p in points]
        return points[int(np.argmax(scores))]

    def _score_point(self, robot_pos, point, world_state):
        ego_dist = np.linalg.norm(point - robot_pos[:2])

        teammate_dist = min(
            np.linalg.norm(point - t.position[:2])
            for t in world_state.get_teammates(self.robot_id, self.is_yellow)
        )

        opponent_dist = min(
            np.linalg.norm(point - o.position[:2])
            for o in world_state.get_opponents(is_yellow=self.is_yellow)
        )

        return -1.0 * ego_dist - 0.5 * teammate_dist + 2.0 * opponent_dist

    # ========================================================
    # Defensive pressure
    # ========================================================
    def _pressure_opponent(self, robot_pos, opp_pos):
        direction = opp_pos[:2] - robot_pos[:2]
        direction = direction / (np.linalg.norm(direction) + 1e-6)

        target = robot_pos[:2] + direction * self.cfg.pressure_distance

        vx, vy = self.movement.go_to(robot_pos, target)
        w = self.movement.face_towards(robot_pos, opp_pos[:2])

        return RobotCommand(
            robot_id=self.robot_id,
            vx=vx,
            vy=vy,
            w=w,
            kick=0,
            dribble=1,
        )

    # ========================================================
    # Helpers
    # ========================================================
    def _go_to_ball(self, robot_pos, ball_pos):
        vx, vy = self.movement.go_to(robot_pos, ball_pos)
        w = self.movement.face_towards(robot_pos, ball_pos)

        return RobotCommand(
            robot_id=self.robot_id,
            vx=vx,
            vy=vy,
            w=w,
            kick=0,
            dribble=1,
        )

    def _trajectory_clear(self, start, end, obstacles, clearance=150):
        for obs in obstacles:
            if self._point_segment_distance(obs.position[:2], start, end) < clearance:
                return False
        return True

    def _point_segment_distance(self, p, a, b):
        ap = p - a
        ab = b - a
        t = np.clip(np.dot(ap, ab) / (np.dot(ab, ab) + 1e-6), 0, 1)
        return np.linalg.norm(ap - t * ab)

    def _generate_grid_points(self, n=50):
        xs = np.linspace(self.field_limits.xmin, self.field_limits.xmax, int(np.sqrt(n)))
        ys = np.linspace(self.field_limits.ymin, self.field_limits.ymax, int(np.sqrt(n)))
        return [np.array([x, y]) for x in xs for y in ys]

    def _select_safest_point(self, world_state):
        points = self._generate_grid_points()
        scores = [
            min(
                np.linalg.norm(p - o.position[:2])
                for o in world_state.get_opponents(is_yellow=self.is_yellow)
            )
            for p in points
        ]
        return points[int(np.argmax(scores))]

    def _pass_kick_power(self, robot_pos, target):
        dist = np.linalg.norm(target - robot_pos[:2])
        return min(self.cfg.max_kick_power, dist * 0.002)


if __name__ == "__main__":
    strategy_config = StrategyConfig(
        possession_distance=200.0,
        possession_angle_threshold=np.radians(30),
        pressure_distance=300.0,
        max_kick_power=10.0,
    )

    field_section = FieldSection(
        xmin=-4500,
        xmax=0,
        ymin=-3000,
        ymax=3000,
    )

    goal_points = [np.array([-4500, 0]), np.array([-4500, 500]), np.array([-4500, -500])]
    strategy = BaseStrategy(
        robot_id=1,
        is_yellow=True,
        section=field_section,
        field_limits=FieldSection(-4500, 4500, -3000, 3000),
        goal_points=goal_points,
        config=strategy_config,
    )
    # The strategy instance can now be used within a control loop
    print("Strategy initialized for robot ID 1 on Yellow team.")
