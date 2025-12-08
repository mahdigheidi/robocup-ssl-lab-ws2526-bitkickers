import math
import time

import numpy as np
from TeamControl.world.transform_cords import world2robot


class PDController:
    """PD controller with trapezoidal velocity profile feed-forward."""

    def __init__(self) -> None:
        self.prev_error: float = 0.0
        self.prev_derivative: float = 0.0
        self.prev_time: float = time.time()

    def compute(
        self,
        error: float,
        max_output: float,
        max_acceleration: float,
        p_gain: float,
        d_gain: float,
        stop_threshold: float,
        d_low_pass: float = 0.5,
    ) -> float:
        """
        Compute control output using PD control with trapezoidal velocity profile.

        Args:
            error: Current error (distance or angle)
            max_output: Maximum output velocity
            max_acceleration: Maximum acceleration for trapezoidal profile
            p_gain: Proportional gain
            d_gain: Derivative gain
            stop_threshold: Error threshold below which output is zero
            d_low_pass: Low-pass filter coefficient for derivative (0-1)

        Returns:
            Control output (velocity)
        """
        delta_time = time.time() - self.prev_time
        self.prev_time = time.time()

        # Prevent division by zero
        if delta_time < 1e-6:
            delta_time = 1e-6

        # Trapezoidal velocity profile feed-forward
        decel_distance = (max_output * max_output) / (2 * max_acceleration)

        if error < 2 * decel_distance:
            v_peak = math.sqrt(2 * max_acceleration * error)
            v_ff = min(v_peak, max_output)
        else:
            v_ff = max_output

        # Derivative with low-pass filter
        derivative = (error - self.prev_error) / delta_time
        derivative = d_low_pass * derivative + (1 - d_low_pass) * self.prev_derivative

        # PD control + feed forward
        output = p_gain * error + d_gain * derivative + v_ff

        # Clamp to max output
        output = max(min(output, max_output), -max_output)

        # Stop if close enough
        if error < stop_threshold:
            output = 0.0

        self.prev_error = error
        self.prev_derivative = derivative

        return output


class RobotMovement:
    """Provides basic movement control utilities for robots."""

    def __init__(self) -> None:
        self.position_controller = PDController()
        self.rotation_controller = PDController()

    def go_to(
        self,
        robot_pos: tuple[float, float, float] | np.ndarray,
        target: tuple[float, float] | np.ndarray,
        kicker_to_target: bool = False,
        max_speed: float = 5000,  # mm/s
        max_acceleration: float = 1700,  # mm/s^2
        kicker_offset: float = 102.0,  # mm
        proportional_gain: float = 0.05,
        differential_gain: float = 0.01,
        differential_low_pass: float = 0.5,
        stop_threshold: float = 1.0,  # mm
    ) -> tuple[float, float]:
        """
        Calculate the velocity (vx, vy) for the robot to move towards a target position.

        Args:
            robot_pos (tuple[float, float, float]): Robot's current position (x, y, theta) in world coordinates.
            target (tuple[float, float]): Target position in world coordinates.
            kicker_to_target: True for kicker to target position, False for robot center to target position.
            max_speed: Maximum linear speed in mm/s
            max_acceleration: Maximum linear acceleration in mm/s^2
            kicker_offset: Offset distance for kicker (mm)
            proportional_gain: P gain for position control
            differential_gain: D gain for position control
            differential_low_pass: Low-pass filter for position derivative
            stop_threshold: Distance threshold below which robot stops (mm)

        Returns:
            tuple[float, float]: Velocity components (vx, vy) in m/s.
        """
        x, y, theta = robot_pos

        if kicker_to_target:
            x += kicker_offset

        # Convert target from world coordinates to robot coordinates
        dx, dy = world2robot((x, y, theta), target)  # type: ignore

        # Calculate position error
        position_error = math.hypot(dx, dy)

        # Use position controller to get desired speed
        desired_v = self.position_controller.compute(
            error=position_error,
            max_output=max_speed,
            max_acceleration=max_acceleration,
            p_gain=proportional_gain,
            d_gain=differential_gain,
            d_low_pass=differential_low_pass,
            stop_threshold=stop_threshold,
        )

        # Convert to x,y components (avoid division by zero)
        if position_error > 1e-6:
            vx_cmd = (dx / position_error) * desired_v
            vy_cmd = (dy / position_error) * desired_v
        else:
            vx_cmd = 0.0
            vy_cmd = 0.0

        # Convert linear velocities from mm/s to m/s
        return vx_cmd / 1000, vy_cmd / 1000

    def face_towards(
        self,
        robot_pos: tuple[float, float, float] | np.ndarray,
        target: tuple[float, float] | np.ndarray,
        max_angular_speed: float = 4.0,  # radians/s
        max_angular_acceleration: float = 5.0,  # rad/s^2
        angular_proportional_gain: float = 0.5,
        angular_differential_gain: float = 0.15,
        angular_differential_low_pass: float = 0.3,
        stop_threshold: float = 0.05,  # radians
    ) -> float:
        """
        Calculate the angular velocity (w) for the robot to face towards a target point.

        Args:
            robot_pos (tuple[float, float, float]): Robot's current position (x, y, theta) in world coordinates.
            target (tuple[float, float]): Target point to face towards in world coordinates.
            max_angular_speed: Maximum angular speed in rad/s
            max_angular_acceleration: Maximum angular acceleration in rad/s^2
            angular_proportional_gain: P gain for rotation control
            angular_differential_gain: D gain for rotation control
            angular_differential_low_pass: Low-pass filter for rotation derivative
            stop_threshold: Angle threshold below which robot stops rotating (radians)

        Returns:
            float: Angular velocity (w) in rad/s.
        """
        x, y, theta = robot_pos

        # Calculate target angle
        target_angle = math.atan2(target[1] - y, target[0] - x)
        theta_error = (target_angle - theta + math.pi) % (2 * math.pi) - math.pi

        # Use rotation controller to get desired angular speed
        w_cmd = self.rotation_controller.compute(
            error=abs(theta_error),
            max_output=max_angular_speed,
            max_acceleration=max_angular_acceleration,
            p_gain=angular_proportional_gain,
            d_gain=angular_differential_gain,
            d_low_pass=angular_differential_low_pass,
            stop_threshold=stop_threshold,
        )

        # Apply correct sign to angular velocity
        w_cmd = w_cmd if theta_error > 0 else -w_cmd

        return w_cmd

    @classmethod
    def moveagent(
        cls,
        robot_pos: tuple[float, float, float],
        target: tuple[float, float],
        velocity: float = 1.0,
    ) -> tuple[float, float]:
        """
        Calculate the velocity (vx, vy) for the robot to move towards a target
        point
        and stop when is very close to it.
        Args:
        robot_pos (tuple[float, float, float]): Robot's current position
        (x, y, theta) in world coordinates.
        target (tuple[float, float]): Target position in world
        coordinates.
        velocity (float, optional): Constant velocity to move towards the
        target. Defaults to 1.0.
        stop_threshold (float, optional): Distance threshold to stop near
        the target. Defaults to 150.
        Returns:
        tuple[float, float]: Velocity components (vx, vy) in robot
        coordinates.
        """
        if robot_pos is None or target is None:
            return 0.0, 0.0
        # Convert target from world coordinates to robot coordinates
        target_robot_frame = world2robot(robot_pos, target)
        assert target_robot_frame is not None
        # Distance to target
        dx, dy = target_robot_frame
        distance = math.sqrt(dx**2 + dy**2)
        # Normalize direction and scale by velocity
        vx, vy = 0.0, 0.0
        if distance > 300:
            vx = (dx / distance) * velocity
            vy = (dy / distance) * velocity
        elif 150 < distance <= 300:
            vx = (dx / distance) * (velocity / 3)
            vy = (dy / distance) * (velocity / 3)
        elif 10 < distance <= 150:
            vx = (dx / distance) * (velocity / 7)
            vy = (dy / distance) * (velocity / 7)

        return vx, vy
