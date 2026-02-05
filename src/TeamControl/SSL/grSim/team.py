# ======================
# TEAM CONTROL MAIN FILE
# ======================

from argparse import ArgumentParser
import re
import time
import sys
import numpy as np
import random
from typing import Optional
from TeamControl.ball.ball_spawner import BallSpawner
from TeamControl.network.ssl_sockets import grSimSender
from TeamControl.ball.send_ball_grsim import send_ball_to_grsim

from TeamControl.world.model import WorldModel
from TeamControl.robot.robot import Robot
from TeamControl.robot.context import WorldState
from TeamControl.robot import enable_logging

from TeamControl.robot.roles.striker import Striker
from TeamControl.robot.roles.support import Support
from TeamControl.robot.roles.defender import Defender
from TeamControl.robot.roles.goalie import Goalkeeper

from TeamControl.robot.roles.passer import Passer

from TeamControl.robot.roles.midfielder import Midfielder

# Collision detection constants
COLLISION_THRESHOLD = 180.0  # mm - robots touching (2 * ROBOT_RADIUS ~177mm)

SIM_IP = "127.0.0.1"
CMD_LISTEN_PORT = 20011
DIVISION = "B"


YELLOW_GOAL = np.array([4500.0, 0.0])

BLUE_GOAL = np.array([-4500.0, 0.0])


def spawn_ball_randomly_in_the_middle():
    x = random.uniform(-0.5, 0.5)
    y = random.uniform(-0.5, 0.5)
    send_ball_to_grsim(x, y, 0, 0)
    time.sleep(0.1)


def check_collisions(world_state: WorldState) -> list[dict]:
    """
    Check for collisions between all robots.
    Returns list of collision events with robot IDs and relative speed.
    """
    collisions = []
    
    # Get all robot states
    yellow_robots = []
    blue_robots = []
    
    for rid in range(6):  # Check robot IDs 0-5
        ys = world_state.get_robot_state(rid, is_yellow=True)
        if ys is not None:
            yellow_robots.append((rid, True, ys))
        bs = world_state.get_robot_state(rid, is_yellow=False)
        if bs is not None:
            blue_robots.append((rid, False, bs))
    
    all_robots = yellow_robots + blue_robots
    
    # Check all pairs
    for i, (rid1, is_yellow1, state1) in enumerate(all_robots):
        for rid2, is_yellow2, state2 in all_robots[i+1:]:
            pos1 = state1.position[:2]
            pos2 = state2.position[:2]
            
            dist = np.linalg.norm(pos1 - pos2)
            
            if dist < COLLISION_THRESHOLD:
                # Calculate relative speed
                vel1 = state1.velocity[:2] if hasattr(state1, 'velocity') and state1.velocity is not None else np.zeros(2)
                vel2 = state2.velocity[:2] if hasattr(state2, 'velocity') and state2.velocity is not None else np.zeros(2)
                relative_speed = np.linalg.norm(vel1 - vel2)
                
                team1 = "Y" if is_yellow1 else "B"
                team2 = "Y" if is_yellow2 else "B"
                
                collisions.append({
                    "robot1": f"{team1}{rid1}",
                    "robot2": f"{team2}{rid2}",
                    "distance": dist,
                    "relative_speed": relative_speed,
                })
    
    return collisions


# MAIN TEAM PROCESS


def team_control_process(wm: WorldModel):
    enable_logging()

    # Determine which teams to run from command line arguments
    # Default is both teams if neither or both are specified
    run_yellow = True
    run_blue = True

    parser = ArgumentParser()
    parser.add_argument("ip", type=str, nargs='?', default="127.0.0.1", help="grSim IP address")
    parser.add_argument("--yellow", action="store_true", help="Control only yellow team")
    parser.add_argument("--blue", action="store_true", help="Control only blue team")
    parser.add_argument("--ball", action="store_true", help="Send ball commands (do not enable in competition!)")

    parsed_args = parser.parse_args()

    # Override SIM_IP if provided
    if parsed_args.ip:
        global SIM_IP
        SIM_IP = parsed_args.ip

    # Determine which teams to run based on parsed arguments
    run_blue = not parsed_args.yellow
    run_yellow = not parsed_args.blue
    run_ball = parsed_args.ball


    if run_yellow:
        print("Starting YELLOW team...")
    if run_blue:
        print("Starting BLUE team...")
    print(f"Sim IP: {SIM_IP}")

    print("------------------------------------")

    # Initialize match standing
    goal_count_blue = 0
    goal_count_yellow = 0

    sender = grSimSender(
        ip=SIM_IP,
        port=CMD_LISTEN_PORT,
        is_yellow=True,
    )
    blue_sender = grSimSender(
        ip=SIM_IP,
        port=CMD_LISTEN_PORT,
        is_yellow=False,
    )

    spawner = BallSpawner(division=DIVISION)

    # Create yellow robots
    robots = {}
    if run_yellow:
        goalie = Robot(robot_id=0, is_yellow=True, role=Goalkeeper())
        defender = Robot(robot_id=1, is_yellow=True, role=Defender(home=(2200.0, -1000.0)))
        midfielder = Robot(robot_id=2, is_yellow=True, role=Midfielder())
        striker = Robot(robot_id=3, is_yellow=True, role=Striker(blue_goal=BLUE_GOAL))

        robots = {
            0: goalie,
            1: defender,
            2: midfielder,
            3: striker,
        }

        # Spawn robots
        goalie.spawn(x=4500, y=0, orientation=np.pi)
        time.sleep(0.05)
        defender.spawn(x=3000, y=0, orientation=0.0)
        time.sleep(0.05)
        midfielder.spawn(x=1000, y=0, orientation=0.0)
        time.sleep(0.05)
        striker.spawn(x=-2500, y=0, orientation=0.0)
        time.sleep(0.05)

        # Initial phases
        striker.switch_phase(Striker.Phase.RECEIVING)
        goalie.switch_phase(Goalkeeper.Phase.ACTIVE)
        defender.switch_phase(Defender.Phase.CHASING)
        midfielder.switch_phase(Midfielder.Phase.CHASING)

    blue_robots = {}
    if run_blue:
        # Create blue robots
        try:
            blue_goalie = Robot(robot_id=0, is_yellow=False, role=Goalkeeper())
            blue_defender = Robot(robot_id=1, is_yellow=False, role=Defender())
            blue_midfielder = Robot(robot_id=2, is_yellow=False, role=Midfielder())
            blue_striker = Robot(robot_id=3, is_yellow=False, role=Striker())

            blue_goalie.spawn(x=-4500, y=0, orientation=0.0)
            time.sleep(0.05)
            blue_defender.spawn(x=-3000, y=0, orientation=0.0)
            time.sleep(0.05)
            blue_midfielder.spawn(x=-2000, y=0, orientation=0.0)
            time.sleep(0.05)
            blue_striker.spawn(x=2500, y=0, orientation=0.0)
            time.sleep(0.05)

            blue_robots = {0: blue_goalie, 1: blue_defender, 2: blue_midfielder, 3: blue_striker}

            # Initial phases
            blue_striker.switch_phase(Striker.Phase.RECEIVING)
            blue_goalie.switch_phase(Goalkeeper.Phase.ACTIVE)
            blue_defender.switch_phase(Defender.Phase.CHASING)
            blue_midfielder.switch_phase(Midfielder.Phase.CHASING)

        except Exception as e:
            print("Blue team spawn failed:", e)

    # Spawn ball
    if run_ball:
        send_ball_to_grsim(0.0, 0.0, 0.0, 0.0)

    # World state
    world_state = WorldState()
    version = 0
    last_time = time.time()
    
    # Collision tracking
    total_collisions = 0
    active_collisions: set[tuple[str, str]] = set()  # Track ongoing collisions to avoid double-counting

    # MAIN LOOP

    while True:
        if version < wm.get_version():
            version = wm.get_version()
            now = time.time()
            last_time = now

            world_state.update(wm, now)

            for rid, robo in robots.items():
                st = world_state.get_robot_state(rid, is_yellow=True)

                if st:
                    robo.update_position(st.position)

            
            # Check for collisions
            collisions = check_collisions(world_state)
            current_collision_pairs = set()
            
            for col in collisions:
                pair = (col["robot1"], col["robot2"])
                current_collision_pairs.add(pair)
                
                # Only count as new collision if wasn't already colliding
                if pair not in active_collisions:
                    total_collisions += 1
                    print(f"⚠️  COLLISION #{total_collisions}: {col['robot1']} <-> {col['robot2']} | "
                          f"dist={col['distance']:.0f}mm, rel_speed={col['relative_speed']:.0f}mm/s")
            
            # Update active collisions
            active_collisions = current_collision_pairs
            for rid, robo in blue_robots.items():
                bt = world_state.get_robot_state(rid, is_yellow=False)
                if bt:
                    robo.update_position(bt.position)

        # Execute behaviors

        for robo in robots.values():
            robo.execute_behavior(
                world_state=world_state,
                sender=sender,
                spawner=spawner,
            )
        for robo in blue_robots.values():
            robo.execute_behavior(
                world_state=world_state,
                sender=blue_sender,
                spawner=spawner,
            )

        # ------------   CHECK FOR GOAL (DO NOT CHANGE)  ------------
        ball_pos = world_state.ball.position
        goal_y_min = -spawner.goal_width / 2
        goal_y_max = spawner.goal_width / 2

        if ball_pos[0] >= spawner.goal_line_x and goal_y_min <= ball_pos[1] <= goal_y_max:
            goal_count_blue += 1
            print(f"Blue Team scored a GOAL !!!!!")
            print(f"Blue Team Goals: {goal_count_blue}")
            print(f"Yellow Team Goals: {goal_count_yellow}")
            if run_ball:
                spawn_ball_randomly_in_the_middle()

        elif ball_pos[0] <= -spawner.goal_line_x and goal_y_min <= ball_pos[1] <= goal_y_max:
            goal_count_yellow += 1
            print(f"Yellow Team scored a GOAL !!!!!")
            print(f"Yellow Team Goals: {goal_count_yellow}")
            print(f"Blue Team Goals: {goal_count_blue}")
            if run_ball:
                spawn_ball_randomly_in_the_middle()

        elif not spawner.is_ball_in_field(ball_pos):
            if run_ball:
                spawn_ball_randomly_in_the_middle()

        # time.sleep(0.02)
