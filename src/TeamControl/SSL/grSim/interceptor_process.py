"""
Interceptor robot behavior for RoboCup SSL simulation.

This module controls a single blue robot that:
1. Captures the ball when free.
2. Shoots toward the opponent goal when possession is secured.
3. Blocks passing lanes when the opponent (yellow) has possession.
4. Intercepts fast-moving balls using future trajectory prediction.
"""

import time
import math
import numpy as np

from TeamControl.network.ssl_sockets import grSimSender
from TeamControl.network.robot_command import RobotCommand
from TeamControl.robot.Movement import RobotMovement
from TeamControl.robot.context import WorldState, POSSESSION_THRESHOLD
from TeamControl.robot.spawn_robot import send_robot_to_grsim


SAFE_DISTANCE = 600.0       
KICK_POWER = 4           
CAPTURE_NEAR_DIST = 200.0     # Start dribbling when closer than this
CAPTURE_HOLD_DIST = 105.0     # Distance considered full possession

DT = 0.016                    # Simulation timestep
STEPS = 30                    # Future steps for interception
EPS = 0.15                    # Timing tolerance for interception

BALL_SPEED_THRESHOLD = 500
MAX_SPEED = 3000.0       

GOAL_X = 4500               
GOAL_Y = 0.0


# Utility Functions

def avoid_collision(rx, ry, tx, ty, y0, y1):
    """
    Adjust target position to avoid collisions with opponent robots.

    Args:
        rx, ry: Current robot position.
        tx, ty: Intended target position.
        y0, y1: Positions of two yellow robots.

    Returns:
        Adjusted (tx, ty) that maintains a safe distance.
    """
    for y in (y0, y1):
        d = math.hypot(tx - y[0], ty - y[1])
        if 5 < d < SAFE_DISTANCE:
            push = (SAFE_DISTANCE - d) / SAFE_DISTANCE
            dx = (tx - y[0]) / d
            dy = (ty - y[1]) / d
            tx += dx * push * 300
            ty += dy * push * 300
    return tx, ty


def perpendicular_block_point(y0, y1, offset=120.0):
    """
    Compute a point perpendicular to the line between two opponents.

    Used to block passing line when opponents have possession.

    Args:
        y0, y1: Positions of two opponent robots.
        offset: Distance from midpoint along perpendicular direction.

    Returns:
        (x, y) block position.
    """
    mx = (y0[0] + y1[0]) / 2
    my = (y0[1] + y1[1]) / 2

    dx = y1[0] - y0[0]
    dy = y1[1] - y0[1]
    length = math.hypot(dx, dy)

    if length < 1e-6:
        return mx, my

    px = -dy / length
    py = dx / length

    return mx + px * offset, my + py * offset

# Main Control Loop

def interceptor_process(wm):
    """
    The robot:
    - Captures and shoots when ball is free.
    - Blocks when opponents have possession.
    - Intercepts fast moving balls using trajectory prediction.

    Args:
        wm: World model interface providing vision frames.
    """
    sender = grSimSender("127.0.0.1", 20011, is_yellow=False)
    world_state = WorldState()
    movement = RobotMovement()

    # Spawn robot at initial position
    send_robot_to_grsim(False, 0, -1500, 0, 0)
    send_robot_to_grsim(team_yellow=False, robot_id=1, x=-5000, y=-5000, orientation=0)

    kick_angle = None

    while True:
        cmd = RobotCommand(0, 0, 0, 0)

        frame = wm.get_latest_frame()
        if frame is None or frame.ball is None:
            sender.send_command(cmd)
            continue

        world_state.update(wm, time.time())

        try:
            ctx = world_state.get_robot_context(0, is_yellow=False)
        except Exception:
            sender.send_command(cmd)
            continue

        # Robot state
        rx, ry, rtheta = ctx.my_state.position

        # Ball state
        ball_x, ball_y = ctx.ball.position
        ball_vx, ball_vy = ctx.ball.velocity
        ball_speed = math.hypot(ball_vx, ball_vy)

        # Opponent possession check
        yellow_has_possession = any(
            np.linalg.norm(op.position[:2] - np.array([ball_x, ball_y]))
            < POSSESSION_THRESHOLD
            for op in ctx.opponents.values()
        )

        # Yellow robot positions
        y0 = world_state.get_robot_state(0, True).position
        y1 = world_state.get_robot_state(1, True).position

        dist_to_ball = math.hypot(ball_x - rx, ball_y - ry)

        # ======================================================
        # 1. BALL FREE → CAPTURE AND SHOOT        
        # ======================================================

        if not yellow_has_possession and dist_to_ball < CAPTURE_NEAR_DIST:

            # Move closer until full control
            if dist_to_ball > CAPTURE_HOLD_DIST:
                vx, vy = movement.go_to(
                    (rx, ry, rtheta),
                    (ball_x, ball_y),
                    kicker_to_target=True,
                    stop_threshold=5
                )
                w = movement.face_towards(
                    (rx, ry, rtheta),
                    (ball_x, ball_y),
                    max_angular_acceleration=7
                )
                cmd = RobotCommand(0, vx, vy, w, dribble=0)

            # Shoot once aligned
            else:
                if kick_angle is None:
                    kick_angle = math.atan2(GOAL_Y - ry, GOAL_X - rx)

                angle_error = (
                    (kick_angle - rtheta + math.pi) % (2 * math.pi)
                ) - math.pi

                if abs(angle_error) > 4:
                    omega = 7.0 * angle_error
                    cmd = RobotCommand(0, 0, 0, omega, dribble=1)
                else:
                    cmd = RobotCommand(
                        0, 0, 0, 0,
                        kick=KICK_POWER,
                        dribble=0
                    )
                    kick_angle = None

        # ======================================================
        # 2. OPPONENT HAS BALL → BLOCK PASS
        # ======================================================
        elif yellow_has_possession or ball_speed < BALL_SPEED_THRESHOLD:

            tx, ty = perpendicular_block_point(y0, y1, offset=20.0)
            tx, ty = avoid_collision(rx, ry, tx, ty, y0, y1)

            vx, vy = movement.go_to((rx, ry, rtheta), (tx, ty))
            w = movement.face_towards(
                (rx, ry, rtheta),
                (ball_x, ball_y)
            )
            cmd = RobotCommand(0, vx, vy, w)

        # ======================================================
        # 3. FAST BALL → INTERCEPTION
        # ======================================================
        else:
            best_target = None
            best_diff = float("inf")

            for k in range(1, STEPS + 1):
                bx_f = ball_x + ball_vx * k * DT
                by_f = ball_y + ball_vy * k * DT

                dist = math.hypot(bx_f - rx, by_f - ry)
                t_robot = dist / MAX_SPEED
                t_ball = k * DT

                diff = abs(t_robot - t_ball)
                if diff < EPS and diff < best_diff:
                    best_target = (bx_f, by_f)
                    best_diff = diff

            if best_target:
                tx, ty = avoid_collision(
                    rx, ry,
                    best_target[0], best_target[1],
                    y0, y1
                )
                vx, vy = movement.go_to((rx, ry, rtheta), (tx, ty))
                w = movement.face_towards(
                    (rx, ry, rtheta),
                    (ball_x, ball_y)
                )
                cmd = RobotCommand(0, vx, vy, w)

        sender.send_command(cmd)
