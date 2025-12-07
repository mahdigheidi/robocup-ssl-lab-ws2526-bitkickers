import math
import heapq
import numpy as np
import sys
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from TeamControl.obstacles.helpers import *

# Field bounds
X_MIN, X_MAX = -4500.0, 4500.0
Y_MIN, Y_MAX = -3000.0, 3000.0

# Cell size
CELL = 50.0
NX = int((X_MAX - X_MIN) / CELL)  # 180
NY = int((Y_MAX - Y_MIN) / CELL)  # 120


def cell_center(ix: int, iy: int) -> Tuple[float, float]:
    """Return center coordinates (x,y) of cell (ix,iy)."""
    x = X_MIN + (ix + 0.5) * CELL
    y = Y_MIN + (iy + 0.5) * CELL
    return x, y


def point_to_cell(x: float, y: float) -> Tuple[int, int]:
    """
    Given world coordinates (x,y), return the closest cell indices (ix, iy).
    Clamps to grid bounds.
    """
    # compute relative offsets
    fx = (x - X_MIN) / CELL - 0.5  # map to index space, rounding will find closest
    fy = (y - Y_MIN) / CELL - 0.5
    ix = int(round(fx))
    iy = int(round(fy))
    ix = max(0, min(NX - 1, ix))
    iy = max(0, min(NY - 1, iy))
    return ix, iy


def mark_forbidden_rect(a, x0: float, x1: float, y0: float, y1: float):
    """Mark cells whose center lies inside rect [x0,x1] x [y0,y1] with value 1000."""
    # Find index range overlapping the rectangle
    # Convert bounds to indices conservatively
    ix0, iy0 = point_to_cell(x0, y0)
    ix1, iy1 = point_to_cell(x1, y1)
    # Ensure correct ordering
    ix_lo, ix_hi = min(ix0, ix1), max(ix0, ix1)
    iy_lo, iy_hi = min(iy0, iy1), max(iy0, iy1)
    for ix in range(ix_lo, ix_hi + 1):
        for iy in range(iy_lo, iy_hi + 1):
            cx, cy = cell_center(ix, iy)
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                a[ix, iy] = 1000.0
    return a


# # 3. Fill out forbidden areas specified
# # 3.a rectangular forbidden area: 3400 <= x <= 4400 , -1100 <= y <= 1100
# mark_forbidden_rect(3400.0, 4400.0, -1100.0, 1100.0)

# # 3.b the four narrow edge strips (assumption explained above)
# # y in (-3000, -2900) and (2900, 3000)
# mark_forbidden_rect(X_MIN, X_MAX, 2900.0, Y_MAX)    # top strip (2900,3000)
# mark_forbidden_rect(X_MIN, X_MAX, Y_MIN, -2900)     # right vertical strip

# # x in (-4500, -4400) and (4400, 4500)
# mark_forbidden_rect(4400, X_MAX, Y_MIN, Y_MAX)  # bottom strip (-3000,-2900)
# mark_forbidden_rect(X_MIN, -4400.0, Y_MIN, Y_MAX)    # left vertical strip


def mark_static_obstacles(a, static_centers: List[Tuple[float, float]], radius: float = ROBOT_RADIUS):
    """
    For each static obstacle center, mark its corresponding cell and adjacent cells with 1000.
    Conservative marking: mark any cell whose center is within (radius + HALF_DIAG).
    """
    thresh = COLLISION_DISTANCE
    thresh2 = thresh * thresh
    for (ox, oy) in static_centers:
        # compute candidate index range to check
        ix_c, iy_c = point_to_cell(ox, oy)
        # compute how many cells to expand in index coordinates
        # expand by ceil(thresh / CELL)
        expand = int(math.ceil(thresh / CELL))
        ix_lo = max(0, ix_c - expand)
        ix_hi = min(NX - 1, ix_c + expand)
        iy_lo = max(0, iy_c - expand)
        iy_hi = min(NY - 1, iy_c + expand)
        for ix in range(ix_lo, ix_hi + 1):
            for iy in range(iy_lo, iy_hi + 1):
                cx, cy = cell_center(ix, iy)
                dx = cx - ox
                dy = cy - oy
                if dx*dx + dy*dy <= thresh2:
                    a[ix, iy] = 1000.0
    return a


def perpendicular_distance_point_to_line(px, py, x1, y1, x2, y2) -> float:
    """
    Perpendicular distance from point (px,py) to infinite line through (x1,y1)-(x2,y2).
    If start==end (zero-length), return distance to point (x1,y1).
    """
    vx = x2 - x1
    vy = y2 - y1
    if abs(vx) < 1e-9 and abs(vy) < 1e-9:
        return math.hypot(px - x1, py - y1)
    # projection length t (not clamped)
    t = ((px - x1)*vx + (py - y1)*vy) / (vx*vx + vy*vy)
    # closest point on infinite line:
    cx = x1 + t*vx
    cy = y1 + t*vy
    return math.hypot(px - cx, py - cy)


def mark_dynamic_influence(a, dynamics: List[Tuple[Tuple[float,float], Tuple[float,float]]],
                           radius: float = ROBOT_RADIUS,
                           inc: float = 200.0):
    """
    For each dynamic obstacle given by (start, goal), increase by `inc` the cells whose center's
    perpendicular distance to the line (start->goal) is less than threshold.
    """
    thresh = COLLISION_DISTANCE
    for (start, goal) in dynamics:
        x1, y1 = start
        x2, y2 = goal
        # conservative bounding box to limit cells to check
        x_lo = min(x1, x2) - thresh
        x_hi = max(x1, x2) + thresh
        y_lo = min(y1, y2) - thresh
        y_hi = max(y1, y2) + thresh
        ix0, iy0 = point_to_cell(x_lo, y_lo)
        ix1, iy1 = point_to_cell(x_hi, y_hi)
        ix_lo = max(0, min(ix0, ix1))
        ix_hi = min(NX - 1, max(ix0, ix1))
        iy_lo = max(0, min(iy0, iy1))
        iy_hi = min(NY - 1, max(iy0, iy1))
        for ix in range(ix_lo, ix_hi + 1):
            for iy in range(iy_lo, iy_hi + 1):
                cx, cy = cell_center(ix, iy)
                d = perpendicular_distance_point_to_line(cx, cy, x1, y1, x2, y2)
                if d <= thresh:
                    # Optionally we can weigh by distance or keep constant:
                    a[ix, iy] += inc
    return a


# 6. Shortest path (Dijkstra) minimizing sum of cell values along the path.
# We'll allow 8-connected movement. Cost to move into neighbour = value of neighbour cell.
# Start cell cost includes its own cell value as well.

NEIGHBORS_8 = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

def dijkstra_min_cost(a, start: Tuple[int,int], goal: Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
    """
    Dijkstra over grid minimizing sum of a[ix,iy] values along the visited cells.
    Avoid cells with a >= 1000 (forbidden) entirely.
    Returns path as list of (ix,iy) from start to goal inclusive or None if unreachable.
    """
    sx, sy = start
    gx, gy = goal
    if a[sx, sy] >= 1000.0 or a[gx, gy] >= 1000.0:
        return None

    dist = np.full((NX, NY), np.inf)
    prev = np.full((NX, NY, 2), -1, dtype=int)
    # Priority queue elements: (cost_so_far, ix, iy)
    pq = []
    dist[sx, sy] = a[sx, sy]  # include starting cell cost
    heapq.heappush(pq, (dist[sx, sy], sx, sy))

    while pq:
        cost, ix, iy = heapq.heappop(pq)
        if cost > dist[ix, iy]:
            continue
        if (ix, iy) == (gx, gy):
            break
        for dx, dy in NEIGHBORS_8:
            nx_ix = ix + dx
            ny_iy = iy + dy
            if nx_ix < 0 or nx_ix >= NX or ny_iy < 0 or ny_iy >= NY:
                continue
            if a[nx_ix, ny_iy] >= 1000.0:
                continue
            # cost to enter neighbor
            new_cost = cost + a[nx_ix, ny_iy]
            if new_cost < dist[nx_ix, ny_iy]:
                dist[nx_ix, ny_iy] = new_cost
                prev[nx_ix, ny_iy] = [ix, iy]
                heapq.heappush(pq, (new_cost, nx_ix, ny_iy))

    if not np.isfinite(dist[gx, gy]):
        return None

    # reconstruct path
    path = []
    curx, cury = gx, gy
    while not (curx == -1 and cury == -1):
        path.append((curx, cury))
        px, py = prev[curx, cury]
        if px == -1 and py == -1:
            break
        curx, cury = int(px), int(py)
    path.reverse()
    return path


# Utilities for converting path cells back to world coords
def cell_to_point(ix: int, iy: int) -> Tuple[float, float]:
    return cell_center(ix, iy)

def preprocess_best_path(start, goal, static_obstacles, dynamic_obstacles):
    a = np.ones((NX, NY), dtype=float)
    a = mark_forbidden_rect(a, 3400.0, 4400.0, -1100.0, 1100.0)

    # 3.b the four narrow edge strips (assumption explained above)
    # y in (-3000, -2900) and (2900, 3000)
    a = mark_forbidden_rect(a, X_MIN, X_MAX, 2900.0, Y_MAX)    # top strip (2900,3000)
    a = mark_forbidden_rect(a, X_MIN, X_MAX, Y_MIN, -2900)     # right vertical strip

    # x in (-4500, -4400) and (4400, 4500)
    a = mark_forbidden_rect(a, 4400, X_MAX, Y_MIN, Y_MAX)  # bottom strip (-3000,-2900)
    a = mark_forbidden_rect(a, X_MIN, -4400.0, Y_MIN, Y_MAX) 

    a = mark_static_obstacles(a, static_obstacles)
    # mark dynamic influences
    a = mark_dynamic_influence(a, dynamic_obstacles)

    start_cell = point_to_cell(*start)
    goal_cell = point_to_cell(*goal)

    path = dijkstra_min_cost(a, start_cell, goal_cell)
    path_world = [cell_to_point(ix, iy) for ix, iy in path]

    return path_world

# --------------------
# Example usage:
# --------------------
# if __name__ == "__main__":
    # # Example static and dynamic obstacles (replace with your data)
    # static_obstacles = [(3270, -700), (2750, -700), (2990, -450), (2700, 0)]
    # dynamic_obstacles = [((1200, 0), (3500, -1700)), ((500, -1500), (2900, -1300))]
    # # mark static obstacles

    

    # print("start_cell, goal_cell = ", start_cell, goal_cell)

    
    # if path is None:
    #     print("No path found (start or goal may be forbidden or unreachable).")
    # else:
    #     print("Path length (cells):", len(path))
    #     # Convert first few cell indices to world coords to show
    #     for (ix, iy) in path:
    #         a[ix][iy] -= 1000
        
    #     print("First 10 waypoints (world coords):", path_world[:10])
    
    # np.set_printoptions(threshold=sys.maxsize)
    # plt.figure(figsize=(6,10))
    # plt.imshow(a.T, origin='lower')  # transpose so x is horizontal, y is vertical
    # plt.colorbar(label="Cell Value")
    # plt.title("Matrix Visualization (Replace 'a' with your matrix)")
    # plt.tight_layout()
    # plt.show()

