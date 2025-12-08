"""Robot class - encapsulates a single robot with its state machine and behaviors."""

import numpy as np
from TeamControl.robot.Movement import RobotMovement
from TeamControl.robot.robot_state_machine import RobotStateMachine, RobotContext, check_possession
from TeamControl.robot.spawn_robot import send_robot_to_grsim
from TeamControl.network.ssl_sockets import grSimSender


class Robot:
    """A robot with autonomous behavior based on roles and state machines."""

    def __init__(self, robot_id: int, is_yellow: bool, initial_role: str = "receiver"):
        """
        Initialize a robot.

        Args:
            robot_id: Robot ID (0-15)
            is_yellow: True if yellow team, False if blue team
            initial_role: Starting role ("passer" or "receiver")
        """
        self.robot_id = robot_id
        self.is_yellow = is_yellow
        self.movement = RobotMovement()
        self.state_machine = RobotStateMachine(initial_role)

        # Current state
        self.position: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.role: str = initial_role

    def spawn(self, x: float, y: float, orientation: float = 0.0):
        """
        Spawn robot in simulator at given position.

        Args:
            x: X position in mm
            y: Y position in mm
            orientation: Orientation in radians
        """
        send_robot_to_grsim(
            team_yellow=self.is_yellow, robot_id=self.robot_id, x=x, y=y, orientation=orientation
        )

    def update_position(self, position: np.ndarray):
        """Update robot's current position from world model."""
        self.position = position

    def update_state(
        self,
        role: str,
        other_robot_pos: np.ndarray,
        ball_pos: np.ndarray,
        ball_velocity: np.ndarray,
    ):
        """
        Update robot's state machine based on world state.

        Args:
            role: Current role ("passer" or "receiver")
            other_robot_pos: Position of the other robot
            ball_pos: Ball position
            ball_velocity: Ball velocity
        """
        self.role = role
        self.state_machine.update(
            robot_id=self.robot_id,
            my_role=role,
            my_pos=self.position,
            my_movement=self.movement,
            other_pos=other_robot_pos,
            ball_pos=ball_pos,
            ball_velocity=ball_velocity,
        )

    def execute_behavior(self, sender: grSimSender, **kwargs):
        """
        Execute the robot's current state behavior.

        Args:
            sender: grSim command sender
            **kwargs: Additional parameters needed by states (e.g., spawner, passing_min_distance)
        """
        # Build context for execution
        context = RobotContext(
            robot_id=self.robot_id,
            my_role=self.role,
            my_pos=self.position,
            my_movement=self.movement,
            other_pos=kwargs.get("other_pos", np.array([0.0, 0.0, 0.0])),
            ball_pos=kwargs.get("ball_pos", np.array([0.0, 0.0])),
            ball_velocity=kwargs.get("ball_velocity", np.array([0.0, 0.0])),
            has_possession=check_possession(
                kwargs.get("ball_pos", np.array([0.0, 0.0])), self.position
            ),
        )

        self.state_machine.execute(context, sender=sender, **kwargs)

    def has_ball(self, ball_pos: np.ndarray, threshold: float = 120) -> bool:
        """
        Check if this robot has possession of the ball.

        Args:
            ball_pos: Ball position
            threshold: Distance threshold for possession (mm)

        Returns:
            True if robot has possession
        """
        return check_possession(ball_pos, self.position, threshold)
