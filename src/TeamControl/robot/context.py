from __future__ import annotations
"""Robot context and world state - shared data structures for robot state machines."""

from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
from typing import Optional


if TYPE_CHECKING:
    from TeamControl.world.model import WorldModel
# Configuration Constants

POSSESSION_THRESHOLD = 120  # mm - distance threshold for ball possession


@dataclass(slots=True, frozen=True)
class EntityState:
    """Position and velocity of an entity (robot or ball).
    
    For robots: position is [x, y, orientation], velocity is [vx, vy, vw]
    For ball: position is [x, y], velocity is [vx, vy]
    """
    position: np.ndarray
    velocity: np.ndarray

    def __post_init__(self):
        """Validate array shapes."""
        if self.position.shape[0] not in (2, 3):
            raise ValueError(
                f"position must have shape (2,) or (3,), got {self.position.shape}"
            )
        if self.velocity.shape[0] not in (2, 3):
            raise ValueError(
                f"velocity must have shape (2,) or (3,), got {self.velocity.shape}"
            )


@dataclass(slots=True)
class RobotContext:
    """Context object containing all information needed for robot state transitions.
    
    This is an immutable snapshot of world state at a single point in time.
    Passed to all state machines during execution.
    """

    robot_id: int
    is_yellow: bool
    my_state: EntityState  # My position and velocity
    ball: EntityState  # Ball position and velocity
    teammates: dict[int, EntityState]  # Other robots on my team (excluding self)
    opponents: dict[int, EntityState]  # All robots on opposing team
    has_possession: bool
    preferred_teammate: Optional[int] = None

    def __post_init__(self):
        """Validate shapes and constraints."""
        # Robot positions should be [x, y, orientation]
        if self.my_state.position.shape != (3,):
            raise ValueError(
                f"my_state.position must have shape (3,), got {self.my_state.position.shape}"
            )
        
        # Ball position should be [x, y]
        if self.ball.position.shape != (2,):
            raise ValueError(
                f"ball.position must have shape (2,), got {self.ball.position.shape}"
            )


@dataclass(frozen=True)
class RoleArea:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    defense_x: float | None = None
    defense_y_half: float | None = None

    def for_team(self, is_yellow: bool) -> "RoleArea":
        if is_yellow:
            return self
        
        xmin = -self.x_max
        xmax = -self.x_min

        return RoleArea(
            x_min=min(xmin, xmax),
            x_max=max(xmin, xmax),
            y_min=self.y_min,
            y_max=self.y_max,
            defense_x=(
                -self.defense_x if self.defense_x is not None else None
            ),
            defense_y_half=self.defense_y_half,
        )

    def is_point_legal(self, context: "RobotContext", x: float, y: float, is_goalie: bool = False) -> bool:
        from TeamControl.obstacles.helpers import defense_area_polygon_for_team
        from shapely.geometry import Point, Polygon
        
        area = self.for_team(context.is_yellow)

        # 1) Rectangle bounds
        if not (
            area.x_min <= x <= area.x_max
            and area.y_min <= y <= area.y_max
        ):
            return False

        # 2) Defense exclusion (unless goalie)
        if not is_goalie:
            # Own defense area
            own_poly = Polygon(defense_area_polygon_for_team(context.is_yellow))
            if Point(x, y).within(own_poly):
                return False
            
            # Opponent defense area
            opp_poly = Polygon(defense_area_polygon_for_team(not context.is_yellow))
            if Point(x, y).within(opp_poly):
                return False

        return True

    def project_to_legal_area(
        self,
        context: "RobotContext",
        x: float,
        y: float,
        is_goalie: bool = False,
    ) -> np.ndarray:
        """
        Projects (x, y) into the nearest legal point inside the role area and within field.
        """
        from TeamControl.obstacles.helpers import defense_area_polygon_for_team, ROBOT_RADIUS
        from shapely.geometry import Point, Polygon
        
        area = self.for_team(context.is_yellow)
        
        # Buffer to keep robot body inside field
        buffer = ROBOT_RADIUS + 10.0 # extra 10mm margin
        
        # 1) Clamp to field boundaries with buffer
        fx_min = -4500.0 + buffer
        fx_max = 4500.0 - buffer
        fy_min = -3000.0 + buffer
        fy_max = 3000.0 - buffer
        
        x = float(np.clip(x, fx_min, fx_max))
        y = float(np.clip(y, fy_min, fy_max))

        # 2) Clamp to role area rectangle
        cx = float(np.clip(x, area.x_min, area.x_max))
        cy = float(np.clip(y, area.y_min, area.y_max))

        # 3) If inside a defense area and not allowed, push out
        if not is_goalie:
            # Check own defense
            own_poly = Polygon(defense_area_polygon_for_team(context.is_yellow))
            if Point(cx, cy).buffer(buffer).intersects(own_poly):
                # Push x away from goal
                if context.is_yellow:
                    cx = min(cx, 3500.0 - buffer) 
                else:
                    cx = max(cx, -3500.0 + buffer)
            
            # Check opponent defense
            opp_poly = Polygon(defense_area_polygon_for_team(not context.is_yellow))
            if Point(cx, cy).buffer(buffer).intersects(opp_poly):
                # Push x away from opponent goal
                if context.is_yellow:
                    cx = max(cx, -3500.0 + buffer)
                else:
                    cx = min(cx, 3500.0 - buffer)

        return np.array([cx, cy], dtype=float)


class WorldState:
    """Single source of truth for world state across all robots.
    
    Tracks state history to compute velocities. Updated once per frame in the main process.
    Provides RobotContext views for individual robots.
    """
    
    def __init__(self):
        """Initialize world state with empty data."""
        # Current state
        self._ball: EntityState = EntityState(
            position=np.zeros(2, dtype=np.float32),
            velocity=np.zeros(2, dtype=np.float32)
        )
        # (is_yellow: bool, robot_id: int) -> EntityState
        self._robots: dict[tuple[bool, int], EntityState] = {}
        
        # History for velocity computation
        self._prev_ball_pos: np.ndarray | None = None
        self._prev_robot_positions: dict[tuple[bool, int], np.ndarray] = {}
        self._prev_timestamp: float | None = None
    
    def update(self, wm: "WorldModel", timestamp: float) -> None:
        """Update world state from world model.
        
        Args:
            wm: WorldModel instance with latest vision data
            timestamp: Current timestamp in seconds
        """
        # Compute time delta
        dt = (timestamp - self._prev_timestamp) if self._prev_timestamp is not None else 0.0
        
        # Update ball
        ball_pos = wm.get_latest_frame().ball.position
        if self._prev_ball_pos is not None and dt > 0:
            ball_vel = (ball_pos - self._prev_ball_pos) / dt
        else:
            ball_vel = np.zeros(2, dtype=np.float32)
        self._ball = EntityState(position=ball_pos, velocity=ball_vel)
        
        # Update robots - collect all visible robots from world model
        new_robots: dict[tuple[bool, int], EntityState] = {}
        
        # Helper to process robots for a team
        for is_yellow in [True, False]:
            for robot_id in range(16):  # SSL allows robot IDs 0-15
                try:
                    robot = wm.get_yellow_robots(isYellow=is_yellow, robot_id=robot_id)
                    if robot is None:
                        continue
                    robot_pos: np.ndarray = robot.position  # type: ignore
                    robot_key = (is_yellow, robot_id)
                    
                    # Compute velocity
                    if robot_key in self._prev_robot_positions and dt > 0:
                        robot_vel = (robot_pos - self._prev_robot_positions[robot_key]) / dt
                    else:
                        robot_vel = np.zeros(3, dtype=np.float32)
                    
                    new_robots[robot_key] = EntityState(position=robot_pos, velocity=robot_vel)
                    self._prev_robot_positions[robot_key] = robot_pos.copy()
                except Exception:
                    # Robot not visible
                    pass
        
        self._robots = new_robots
        
        # Store for next frame
        self._prev_ball_pos = ball_pos.copy()
        self._prev_timestamp = timestamp
    
    def get_robot_context(self, robot_id: int, is_yellow: bool) -> "RobotContext":
        """Build RobotContext for a specific robot.
        
        Args:
            robot_id: Robot ID (0-15)
            is_yellow: True for yellow team, False for blue team
            
        Returns:
            RobotContext containing world view for this robot
        """
        my_key = (is_yellow, robot_id)
        my_state = self._robots.get(my_key)
    
        
        if my_state is None:
            raise ValueError(f"Robot {robot_id} ({'yellow' if is_yellow else 'blue'}) not found in world state")
        
        # Build teammates dict (same team, excluding self)
        teammates = {
            rid: state 
            for (team_is_yellow, rid), state in self._robots.items()
            if team_is_yellow == is_yellow and rid != robot_id
        }
        
        # Build opponents dict (opposite team)
        opponents = {
            rid: state
            for (team_is_yellow, rid), state in self._robots.items()
            if team_is_yellow != is_yellow
        }
        
        # Check possession
        dist = float(np.linalg.norm(self._ball.position - my_state.position[:2]))
        has_possession = dist < POSSESSION_THRESHOLD
        
        return RobotContext(
            robot_id=robot_id,
            is_yellow=is_yellow,
            my_state=my_state,
            ball=self._ball,
            teammates=teammates,
            opponents=opponents,
            has_possession=has_possession,
        )
    
    @property
    def ball(self) -> EntityState:
        """Get current ball state."""
        return self._ball
    
    def get_robot_state(self, robot_id: int, is_yellow: bool) -> EntityState | None:
        """Get state of a specific robot, or None if not visible."""
        return self._robots.get((is_yellow, robot_id))

