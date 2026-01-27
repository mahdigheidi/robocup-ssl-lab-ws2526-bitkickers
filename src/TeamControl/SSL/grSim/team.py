# ======================
# TEAM CONTROL MAIN FILE
# ======================

import time
import numpy as np

from TeamControl.ball.ball_idle_spawner import BallIdleSpawner
from TeamControl.network.ssl_sockets import grSimSender
from TeamControl.ball.send_ball_grsim import send_ball_to_grsim

from TeamControl.world.model import WorldModel
from TeamControl.robot.robot import Robot
from TeamControl.robot.context import WorldState
from TeamControl.robot import enable_logging

from TeamControl.robot.roles.striker import Striker
from TeamControl.robot.roles.support import Support
from TeamControl.robot.roles.Defender import Defender
from TeamControl.robot.roles.goalie import Goalkeeper


SIM_IP = "127.0.0.1"
CMD_LISTEN_PORT = 20011
DIVISION = "B"


YELLOW_GOAL = np.array([4500.0, 0.0])
BLUE_GOAL = np.array([-4500.0, 0.0])


# MAIN TEAM PROCESS

def team_control_process(wm: WorldModel):
    enable_logging()

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


    spawner = BallIdleSpawner(division=DIVISION)

    # Create robots 
   
    goalie = Robot(
        robot_id=0,
        is_yellow=True,
        role=Goalkeeper(),
    )

    defender = Robot(
        robot_id=1,
        is_yellow=True,
        role=Defender(home=(2200.0, -1000.0)),
    )

    support = Robot(
        robot_id=2,
        is_yellow=True,
        role=Support(yellow_goal=YELLOW_GOAL),
    )

    striker = Robot(
        robot_id=3,
        is_yellow=True,
        role=Striker(blue_goal=BLUE_GOAL),
    )

    robots = {
        0: goalie,
        1: defender,
        2: support,
        3: striker,
    }

    # Spawn robots
  
    goalie.spawn(x=4500, y=0, orientation=np.pi)
    time.sleep(0.05)

    defender.spawn(x=2500, y=0, orientation=0.0)
    time.sleep(0.05)

    support.spawn(x=0, y=0, orientation=0.0)
    time.sleep(0.05)

    striker.spawn(x=-2500, y=0, orientation=0.0)
    time.sleep(0.05)

    # Spawn blue goalie for testing purposes
    try:
        blue_goalie = Robot(
            robot_id=0,
            is_yellow=False,
            role=Goalkeeper(),
        )
        blue_goalie.spawn(x=-4500, y=0, orientation=0.0)
        blue_robots = {0: blue_goalie}
        time.sleep(0.05)
    except Exception:
        pass

    # Spawn ball
    send_ball_to_grsim(0.0, 0.0, 0.0, 0.0)
    time.sleep(0.05)

    # World state
   
    world_state = WorldState()
    version = 0
    last_time = time.time()

    # Initial phases
 
    goalie.switch_phase(Goalkeeper.Phase.ACTIVE)
    defender.switch_phase(Defender.Phase.HOLD)
    support.switch_phase(Support.Phase.BASIC)
    striker.switch_phase(Striker.Phase.BASIC)
    blue_goalie.switch_phase(Goalkeeper.Phase.ACTIVE)


    # MAIN LOOP
  
    while True:
        if version < wm.get_version():
            version = wm.get_version()
            now = time.time()
            last_time = now

            world_state.update(wm, now)

            for rid, robo in robots.items():
                st = world_state.get_robot_state(rid, is_yellow=True)
                bt = world_state.get_robot_state(rid, is_yellow=False)
                if st:
                    robo.update_position(st.position)
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

        #time.sleep(0.02)
