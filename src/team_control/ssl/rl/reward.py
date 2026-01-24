import math
import random
from typing import List, Tuple

from team_control.ssl.vision.robots import Robot
from team_control.ssl.vision.ball import Ball
from team_control.ssl.rl.field import Field
from team_control.ssl.rl.auxiliary import (
    distance,
    COLLISION_DISTANCE,
    is_goal_scored,
    point_in_defense_area,
    is_in_field,
)


TEAM_SPIRIT = 0.7
AGGRESSION = 1.0

# The scale of the rewards should not be too high, if too high the network loses plasiticity and doesn't converge.

# Field Player Rewards
# # Terminating
GOAL_SCORED_REWARD = +120
GOAL_CONCEDED_PENALTY = -100
BALL_OUT_OF_FIELD_PENALTY = -20
# # Non-terminating
DEFENSE_AREA_VIOLATION_PENALTY = -25
ROBOT_IDLE_PENALTY = -10
ROBOT_OUT_OF_FIELD_PENALTY = -80
BALL_IDLE_PENALTY = -10
DISTANCE_TO_BALL_COEF_PENALTY = -50
FACING_BALL_REWARD = +10
COLLISION_PENALTY = -75
# HIGH_SPEED_COLLISION_PENALTY = -30.0
# LOW_SPEED_COLLISION_PENALTY = -5.0

# Goalie Rewards
# # Terminating
GOAL_CONCEDED_PENALTY = -150
# # Non-terminating
DISTANCE_TO_GOAL_COEF_PENALTY = -50
OUTSIDE_DEFENSE_AREA_PENALTY = -25
SAVE_REWARD = +50


class Reward:
    def __init__(self, role, debug_on=False):
        self.role = role
        self.robot: Robot = None
        self.ball: Ball = None
        self.teamates: List[Robot] = None
        self.opponent: List[Robot] = None
        self.prev_robot: Robot = None
        self.prev_ball: Ball = None
        self.prev_teammates: List[Robot] = None
        self.prev_opponents: List[Robot] = None
        self.debug_on = debug_on

    def calculate_reward(
            self,
            field: Field,
            is_team_yellow: bool,
            robot: Robot,
            ball: Ball,
            teammates: List[Robot],
            opponents: List[Robot]):
        if self.role == 0:
            # print("The role is goalkeeper")
            reward, done, info = self._calculate_goalie_reward(field, is_team_yellow, robot, ball, teammates, opponents)
        else:
            reward, done, info = self._calculate_player_reward(field, is_team_yellow, robot, ball, teammates, opponents)

        self.prev_robot = robot
        self.prev_ball = ball
        self.prev_teammates = teammates
        self.prev_opponents = opponents

        if self.debug_on:
            print(info)

        return reward, done

    def _calculate_player_reward(
            self,
            field: Field,
            is_team_yellow: bool,
            robot: Robot,
            ball: Ball,
            teammates: List[Robot],
            opponents: List[Robot]) -> Tuple[float, bool, dict]:

        if not is_team_yellow:
            pass
            # reverse_the_data()

        x, y, o = robot.position
        x_ball, y_ball = ball.position

        x_min, x_max = field.x_max, field.x_min
        y_min, y_max = field.y_max, field.y_min
        goal_line, goal_width = field.x_max, field.goal_width
        penalty_width, penalty_depth = field.penalty_width, field.penalty_depth

        components = {
            # Terminating
            "goal": 0.0,
            "concede_goal": 0.0,
            "ball_out_of_field": 0.0,
            # Non-terminating
            "defense_area_violation": 0.0,
            "robot_idle": 0.0,
            "robot_out_of_field": 0.0,
            "ball_possession": 0.0,
            "ball_idle": 0.0,
            "distance_to_ball": 0.0,
            "facing_ball": 0.0,
            "collision": 0.0,
        }

        # Terminating rewards
        # Goal
        if is_goal_scored(x_ball, y_ball, -goal_line, goal_width):
            components["goal"] = GOAL_SCORED_REWARD
            return GOAL_SCORED_REWARD, True, components

        # Concede goal
        if is_goal_scored(x_ball, y_ball, goal_line, goal_width):
            # print(x_ball, y_ball, -goal_line, goal_width)
            # print("H" * 100)
            components["concede"] = GOAL_CONCEDED_PENALTY
            return GOAL_CONCEDED_PENALTY, True, components

        # Ball out of field
        if not is_in_field(field.div_name, ball.position):
            components["ball_out_of_field"] = BALL_OUT_OF_FIELD_PENALTY
            return BALL_OUT_OF_FIELD_PENALTY, True, components

        # Non-terminating rewards
        # Defense area violation
        if point_in_defense_area('B_Yellow', robot.position) or point_in_defense_area('B_Blue', robot.position):
            components["defense_violation"] = DEFENSE_AREA_VIOLATION_PENALTY

        # Robot idle
        if self.prev_robot is not None and abs(robot.position[0] - self.prev_robot.position[0]) + abs(robot.position[1] - self.prev_robot.position[1]) < 10:
            components["robot_idle"] = ROBOT_IDLE_PENALTY

        # Robot out of field
        if not is_in_field(field.div_name, robot.position):
            components["robot_out_of_field"] = ROBOT_OUT_OF_FIELD_PENALTY

        # Ball idle
        if self.prev_ball is not None and abs(ball.position[0] - self.prev_ball.position[0]) + abs(ball.position[1] - self.prev_ball.position[1]) < 10:
            components["ball_idle"] = BALL_IDLE_PENALTY

        # Distance to ball
        field_diagonal = math.hypot(2*field.x_max, 2*field.y_max)
        distance_to_ball = distance(robot.position, ball.position) / field_diagonal
        components["distance_to_ball"] = DISTANCE_TO_BALL_COEF_PENALTY * distance_to_ball

        # Facing ball
        angle_to_ball = math.atan2(ball.position[1] - robot.position[1], ball.position[0] - robot.position[0]) / math.pi
        robot_angle = robot.position[2] / math.pi
        facing_ball_penalty = abs(angle_to_ball-robot_angle) * 10
        components["facing_ball"] -= facing_ball_penalty

        # Bot collision
        if teammates is not None:
            for teammate in teammates:
                if isinstance(teammate, Robot) and distance(robot.position, teammate.position) <= COLLISION_DISTANCE:
                    components["collision"] = COLLISION_PENALTY
        if opponents is not None:
            for opponent in opponents:
                if isinstance(opponent, Robot) and distance(robot.position, opponent.position) <= COLLISION_DISTANCE:
                    components["collision"] -= COLLISION_PENALTY

        cumulative_reward = 0
        for r in components.values():
            cumulative_reward += r
        return cumulative_reward, False, components

    def _calculate_goalie_reward(
            self,
            field: Field,
            is_team_yellow: bool,
            robot: Robot,
            ball: Ball,
            teammates: List[Robot],
            opponents: List[Robot]) -> Tuple[float, bool, dict]:
        components = {
            # Terminating
            "goal_conceded": 0.0,
            # Non-terminating
            "outside_defense_area": 0.0,
            "goal_saved": 0.0,
            "distance_to_goal": 0.0,
            # "ball_facing": 0.0,
        }
        x_min, x_max = field.x_max, field.x_min
        y_min, y_max = field.y_max, field.y_min
        goal_line, goal_width = field.x_max if is_team_yellow else field.x_min, field.goal_width
        penalty_width, penalty_depth = field.penalty_width, field.penalty_depth
        if is_team_yellow:
            goal_line *= -1

        # Terminating
        # # Goal conceded
        x_ball, y_ball = ball.position
        if is_goal_scored(x_ball, y_ball, goal_line, goal_width):
            components["goal_conceded"] = GOAL_CONCEDED_PENALTY
            return GOAL_CONCEDED_PENALTY, True, components

        # Non-terminating
        # # Outside defense area
        defense_area_name = 'B_' + ('Yellow' if is_team_yellow else 'Blue')
        if not point_in_defense_area(defense_area_name, robot.position):
            components["defense_violation"] = DEFENSE_AREA_VIOLATION_PENALTY

        # # Goal saved
        # If ball is inside defense area and moving toward goal, reward save
        if point_in_defense_area(defense_area_name, robot.position) and point_in_defense_area(defense_area_name, ball.position):
            components["goal_saved"] = SAVE_REWARD

        # distance to the goal
        field_diagonal = math.hypot(2*field.x_max, 2*field.y_max)
        distance_to_goal = distance(robot.position, (goal_line, 0)) / field_diagonal
        if not point_in_defense_area(defense_area_name, robot.position):
            components["distance_to_goal"] = distance_to_goal * DISTANCE_TO_GOAL_COEF_PENALTY

        # # Ball facing
        # pass

        cumulative_reward = 0
        for r in components.values():
            cumulative_reward += r

        return cumulative_reward, False, components
