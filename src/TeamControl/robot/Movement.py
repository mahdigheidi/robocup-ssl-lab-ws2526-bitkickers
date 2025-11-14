import math
from TeamControl.world.transform_cords import world2robot

class RobotMovement:
    """Provides basic movement control utilities for robots."""
    prev_speed = 0.0  

    @classmethod
    def goToPoint(
        cls, 
        robot_pos: tuple[float, float, float], 
        target: tuple[float, float],
        face_ball: bool = False,
        ball_pos: tuple[float, float] = (0.0, 0.0),

        max_g = 3.0,
        min_g = 0.3,
        max_speed: float = 5.0, # mm/s      
        slow_radius=1800.0,

        ) -> tuple[float, float, float]:

        """
        Calculate the velocity (vx, vy) for the robot to move towards a target point
        and stop when it reaches it.

        Args:
            robot_pos (tuple[float, float, float]): Robot's current position (x, y, theta) in world coordinates.
            target (tuple[float, float]): Target position in world coordinates.
            velocity (float, optional): Constant velocity to move towards the target. Defaults to 1.0.

        Returns:
            tuple[float, float, float]: Velocity components (vx, vy, w) in robot coordinates.
        """

        if robot_pos is None or target is None:
            return 0.0, 0.0, 0.0

        x, y, theta = robot_pos

        # Convert target from world coordinates to robot coordinates
        dx, dy = world2robot(robot_pos, target)
        
        kicker_offset = 0
        dx -= kicker_offset 
        distance = math.hypot(dx, dy)

        if distance <= 1e-6:
            desired_speed = 0.0

        else:
            desired_speed = max_speed * (1 - math.exp(-distance / slow_radius))

        cls.prev_speed = desired_speed

        # Convert speed to vx, vy
        if distance > 1e-6:
            vx = (dx / distance) * desired_speed
            vy = (dy / distance) * desired_speed
        else:
            vx = 0.0
            vy = 0.0

        # Rotate kicker to face target
        D_theta = math.atan2(target[1] - y, target[0] - x)
        if face_ball:
            D_theta = math.atan2(ball_pos[1] - y, ball_pos[0] - x)
        theta_error = (D_theta - theta + math.pi) % (2 * math.pi) - math.pi
        
        # logistic model
        k = 0.005
        x0 = 100.0 
        #dist = float(f"{distance:.2f}")
        theta_gain = min_g + (max_g - min_g) / (1 + math.exp(-k * (distance - x0)))

        if distance<=1e-6:
            w = 0.0
        else:
            w = theta_gain * theta_error

        #print(f"RobotMovemet.goToPoint: dx={dx:.2f}, dy={dy:.2f}, distance={distance:.2f}, vx={vx:.2f}, vy={vy:.2f}, w={w:.2f}")

        return vx, vy, w
 