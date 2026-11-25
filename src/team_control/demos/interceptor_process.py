"""
Interceptor robot behavior

This module controls a single blue robot that:
1. Captures and shoots when the ball is free.
2. Blocks passing lanes when the opponent (yellow) has possession
   or the ball is moving slowly.
3. Intercepts fast-moving balls using future trajectory prediction.

"""

import math
import time

import numpy as np

from team_control.behavior.context import POSSESSION_THRESHOLD, WorldState
from team_control.motion.movement import RobotMovement
from team_control.motion.obstacles.helpers import (
    COLLISION_DISTANCE,
    distance,
    is_in_field,
    point_in_defense_area,
)
from team_control.network.robot_command import RobotCommand
from team_control.network.ssl_sockets import grSimSender
from team_control.sim.spawn_robot import send_robot_to_grsim

SAFE_DISTANCE = 600.0          
KICK_POWER = 5               

CAPTURE_NEAR_DIST = 400.0      
CAPTURE_HOLD_DIST = 105.0     

DT = 0.016                  
STEPS = 30              

BALL_SPEED_THRESHOLD = 500.0   

MAX_SPEED = 3000.0             
MAX_ACCEL = 1700.0           

GOAL_X = 4500                
GOAL_Y = 0.0            




def time_to_reach(d_mm: float,
                  v_max: float = MAX_SPEED,
                  a_max: float = MAX_ACCEL) -> float:
    """
    Analytic time estimate for the robot to reach a target distance.

    Uses a trapezoidal (or triangular) velocity profile consistent
    with the movement controller.

    Args:
        d_mm: Distance to target in millimeters
        v_max: Maximum linear speed (mm/s)
        a_max: Maximum linear acceleration (mm/s^2)

    Returns:
        Estimated time in seconds
    """
    d_switch = (v_max * v_max) / a_max

    if d_mm <= d_switch:
      
        return 2.0 * math.sqrt(d_mm / a_max)
    else:
 
        return (d_mm / v_max) + (v_max / a_max)


def safe_target(tx: float, ty: float,
                rx: float, ry: float) -> tuple[float, float]:
    if not is_in_field((tx, ty)):
        return rx, ry
    if point_in_defense_area((tx, ty)):
        return rx, ry
    return tx, ty


def is_safe_from_robots(tx: float, ty: float,
                        robots: list,
                        margin: float = 200.0) -> bool:
    for r in robots:
        if distance((tx, ty), r[:2]) < COLLISION_DISTANCE + margin:
            return False
    return True


def avoid_collision(rx: float, ry: float,
                    tx: float, ty: float,
                    y0, y1) -> tuple[float, float]:
    for y in (y0, y1):
        d = math.hypot(tx - y[0], ty - y[1])
        if 5.0 < d < SAFE_DISTANCE:
            push = (SAFE_DISTANCE - d) / SAFE_DISTANCE
            dx = (tx - y[0]) / d
            dy = (ty - y[1]) / d
            tx += dx * push * 500.0
            ty += dy * push * 500.0
    return tx, ty


def block_pass_point(passer, receiver,
                     offset: float = 20.0) -> tuple[float, float]:
    """
    Compute a blocking point perpendicular to the pass line.

    Args:
        passer: Robot currently closest to the ball
        receiver: Other opponent robot
        offset: Perpendicular offset distance (mm)

    Returns:
        Blocking point (x, y)
    """
    px, py = passer[:2]
    rx, ry = receiver[:2]

    dx = rx - px
    dy = ry - py
    length = math.hypot(dx, dy)

    if length < 1e-6:
        return px, py

    nx = -dy / length
    ny = dx / length
    mx = (px + rx) / 2.0
    my = (py + ry) / 2.0

    return mx + nx * offset, my + ny * offset


# =====================================================
# MAIN CONTROL LOOP
# =====================================================

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

    # Spawn interceptor robot
    send_robot_to_grsim(False, 0, -1500, 0, 0)

    kick_angle = None

    while True:
        cmd = RobotCommand(0, 0, 0, 0)

        # World state update
   
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

        # Opponent states
        y0 = world_state.get_robot_state(0, True).position
        y1 = world_state.get_robot_state(1, True).position

        # Possession check
        yellow_has_possession = any(
            np.linalg.norm(op.position[:2] - np.array([ball_x, ball_y]))
            < POSSESSION_THRESHOLD
            for op in ctx.opponents.values()
        )

        # Identify passer and receiver
        d0 = np.linalg.norm(np.array(y0[:2]) - np.array([ball_x, ball_y]))
        d1 = np.linalg.norm(np.array(y1[:2]) - np.array([ball_x, ball_y]))

        passer = y0 if d0 < d1 else y1
        receiver = y1 if passer is y0 else y0

        dist_to_ball = math.hypot(ball_x - rx, ball_y - ry)

        # =================================================
        # 1. FREE BALL -> CAPTURE & SHOOT
        # =================================================
        if not yellow_has_possession and dist_to_ball < CAPTURE_NEAR_DIST:

            if dist_to_ball > CAPTURE_HOLD_DIST:
                tx, ty = safe_target(ball_x, ball_y, rx, ry)
                vx, vy = movement.go_to(
                    (rx, ry, rtheta),
                    (tx, ty),
                    kicker_to_target=True,
                )
                w = movement.face_towards((rx, ry, rtheta), (tx, ty))
                cmd = RobotCommand(0, vx, vy, w, dribble=0)

            else:
                if kick_angle is None:
                    kick_angle = math.atan2(GOAL_Y - ry, GOAL_X - rx)

                angle_error = (
                    (kick_angle - rtheta + math.pi) % (2 * math.pi)
                ) - math.pi

                if abs(angle_error) > 0.5:
                    cmd = RobotCommand(
                        0, 0, 0, 6.0 * angle_error, dribble=1
                    )
                else:
                    cmd = RobotCommand(0, 0, 0, 0, kick=KICK_POWER)
                    kick_angle = None

        # =================================================
        # 2. OPPONENT HAS BALL OR SLOW BALL -> BLOCK PASS
        # =================================================
        elif yellow_has_possession or ball_speed < BALL_SPEED_THRESHOLD:
            tx, ty = block_pass_point(passer, receiver)
            tx, ty = avoid_collision(rx, ry, tx, ty, y0, y1)
            tx, ty = safe_target(tx, ty, rx, ry)

            vx, vy = movement.go_to((rx, ry, rtheta), (tx, ty), max_acceleration=200)
            w = movement.face_towards((rx, ry, rtheta), (ball_x, ball_y))
            cmd = RobotCommand(0, vx, vy, w)

        # =================================================
        # 3. FAST BALL -> INTERCEPT
        # =================================================
        else:
            best_target = None
            best_diff = float("inf")

            for k in range(1, STEPS + 1):
                bx_f = ball_x + ball_vx * k * DT
                by_f = ball_y + ball_vy * k * DT

                if not is_safe_from_robots(bx_f, by_f, [y0, y1]):
                    continue

                dist_mm = math.hypot(bx_f - rx, by_f - ry)
                t_robot = time_to_reach(dist_mm)
                t_ball = k * DT

                diff = abs(t_robot - t_ball)
                if diff < best_diff:
                    best_diff = diff
                    best_target = (bx_f, by_f)

            if best_target is not None:
                tx, ty = avoid_collision(
                    rx, ry, best_target[0], best_target[1], y0, y1
                )
                tx, ty = safe_target(tx, ty, rx, ry)

                vx, vy = movement.go_to((rx, ry, rtheta), (tx, ty))
                w = movement.face_towards((rx, ry, rtheta), (ball_x, ball_y))
                cmd = RobotCommand(0, vx, vy, w)

        sender.send_command(cmd)
