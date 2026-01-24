from multiprocessing import Process, Queue

from team_control.world.model_manager import WorldModelManager
from team_control.world.model_runner import wm_runner
from team_control.ssl.vision.process import vision_worker
from team_control.ssl.offpolicy_rl.agent import agent_process


def main():
    use_sim = True
    sim_ip = "127.0.0.1"
    cmd_port = 20011
    vision_port = 10020
    is_team_yellow = True
    division = 'B'

    vision_q = Queue()
    gc_q = Queue()  # GameController queue, not in use

    vision_wkr = Process(target=vision_worker, args=(vision_q, use_sim, vision_port,))

    with WorldModelManager() as wm_manager:
        wm = wm_manager.WorldModel()  # pylint: disable=no-member
        wmr = Process(target=wm_runner, args=(wm, vision_q, gc_q,))

        vision_wkr.start()
        wmr.start()

        # For self-play we also create and play the opponent team
        # For main competition it should be commented out
        goalie_us = Process(target=agent_process, args=(wm, division, sim_ip, cmd_port, 0, is_team_yellow, 0))
        goalie_them = Process(target=agent_process, args=(wm, division, sim_ip, cmd_port, 0, not is_team_yellow, 0))
        goalie_us.start()
        goalie_them.start()

        team: list[Process] = []
        opp: list[Process] = []
        for i in range(3):
            teammate = Process(target=agent_process, args=(wm, division, sim_ip, cmd_port, i+1, is_team_yellow, 1))
            opponent = Process(target=agent_process, args=(wm, division, sim_ip, cmd_port, i+1, not is_team_yellow, 1))
            team.append(teammate)
            opp.append(opponent)
            teammate.start()
            opponent.start()

        goalie_us.join()
        goalie_them.join()
        for i in range(3):
            team[i].join()
            opp[i].join()

        vision_wkr.join()
        wmr.join()


if __name__ == "__main__":
    main()
