"""Robot context - shared data structure for robot state machines."""

from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from TeamControl.robot.Movement import RobotMovement


@dataclass(slots=True)
class RobotContext:
    """Context object containing all information needed for robot state transitions.
    
    Uses slots for better memory efficiency and performance.
    This context is passed to all state machines during execution.
    """

    robot_id: int
    my_pos: np.ndarray
    my_movement: "RobotMovement"
    other_pos: np.ndarray  # Position of other robot (for passing, coordination)
    ball_pos: np.ndarray
    ball_velocity: np.ndarray
    has_possession: bool

    def __post_init__(self):
        """Validate context after initialization."""
        if not isinstance(self.robot_id, int) or self.robot_id < 0:
            raise ValueError(f"Invalid robot_id: {self.robot_id}")


def check_possession(ball_pos: np.ndarray, robot_pos: np.ndarray, threshold: float = 120) -> bool:
    """Check if robot has possession of the ball based on distance threshold."""
    dist = float(np.linalg.norm(ball_pos - robot_pos[:2]))
    return dist != 0 and dist < threshold
