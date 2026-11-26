# ======================
# TEAM CONTROL MAIN FILE
# ======================

import random
import time
from argparse import ArgumentParser

import numpy as np

from team_control.behavior import enable_logging
from team_control.behavior.context import WorldState
from team_control.behavior.robot import Robot
from team_control.behavior.roles.defender import Defender
from team_control.behavior.roles.goalie import Goalkeeper
from team_control.behavior.roles.midfielder import Midfielder
from team_control.behavior.roles.striker import Striker
from team_control.network.ssl_sockets import grSimSender
from team_control.sim.ball_spawner import BallSpawner
from team_control.sim.send_ball_grsim import send_ball_to_grsim
from team_control.world.model import WorldModel

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


# MAIN TEAM PROCESS


def team_control_process(wm: WorldModel):
    enable_logging()

    # Determine which teams to run from command line arguments
    # Default is both teams if neither or both are specified
    run_yellow = True
    run_blue = True

    parser = ArgumentParser()
    parser.add_argument("ip", type=str, nargs="?", default="192.168.178.2", help="grSim IP address")
    parser.add_argument("--yellow", action="store_true", help="Control only yellow team")
    parser.add_argument("--blue", action="store_true", help="Control only blue team")
    parser.add_argument(
        "--ball", action="store_true", help="Send ball commands (do not enable in competition!)"
    )

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
    robots: dict[int, Robot] = {}
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

    blue_robots: dict[int, Robot] = {}
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
    time.sleep(2)

    # MAIN LOOP

    while True:
        if version < wm.get_version():
            version = wm.get_version()
            now = time.time()

            world_state.update(wm, now)

            for rid, robo in robots.items():
                try:
                    st = world_state.get_robot_state(rid, is_yellow=True)
                except Exception:
                    continue

                if st:
                    robo.update_position(st.position)

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
            print("Blue Team scored a GOAL !!!!!")
            print(f"Blue Team Goals: {goal_count_blue}")
            print(f"Yellow Team Goals: {goal_count_yellow}")
            if run_ball:
                spawn_ball_randomly_in_the_middle()

        elif ball_pos[0] <= -spawner.goal_line_x and goal_y_min <= ball_pos[1] <= goal_y_max:
            goal_count_yellow += 1
            print("Yellow Team scored a GOAL !!!!!")
            print(f"Yellow Team Goals: {goal_count_yellow}")
            print(f"Blue Team Goals: {goal_count_blue}")
            if run_ball:
                spawn_ball_randomly_in_the_middle()

        elif not spawner.is_ball_in_field(ball_pos):
            if run_ball:
                spawn_ball_randomly_in_the_middle()
