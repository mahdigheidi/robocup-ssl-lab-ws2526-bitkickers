from team_control.motion.obstacles.helpers import (
    COLLISION_DISTANCE,
    distance,
    is_in_field,
    point_in_defense_area,
)

SAFE_MARGIN = 200.0


def constrain_target(
    desired: tuple[float, float],
    robot_pos: tuple[float, float],
    other_robots: list[tuple[float, float]],
    *,
    allow_defense_area: bool = False,
) -> tuple[float, float]:
    """
    Project a desired target to the nearest legal position.
    """

    tx, ty = desired
    rx, ry = robot_pos

    # ---- Field bounds ----
    if not is_in_field((tx, ty)):
        return rx, ry

    # ---- Defense area (unless goalie) ----
    if not allow_defense_area and point_in_defense_area((tx, ty)):
        return rx, ry

    # ---- Robot-robot collision safety ----
    for r in other_robots:
        if distance((tx, ty), r[:2]) < COLLISION_DISTANCE + SAFE_MARGIN:
            return rx, ry

    return tx, ty


def avoid_defense_area_target(
    tx: float,
    ty: float,
    fallback_x: float,
    margin: float = 150.0,
) -> tuple[float, float]:
    """
    If target lies inside opponent defense area, push it safely outside.
    """
    if point_in_defense_area((tx, ty)):
        # Push back along x-axis (away from opponent goal)
        tx = fallback_x - margin
    return tx, ty
