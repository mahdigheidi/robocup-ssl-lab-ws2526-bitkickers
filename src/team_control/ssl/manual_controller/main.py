from multiprocessing import Process, Queue
import time

from team_control.ssl.vision.process import vision_worker
from team_control.world.model_manager import WorldModelManager
from team_control.world.model_runner import wm_runner
from team_control.ssl.manual_controller.striker_process import striker_process
from team_control.ssl.manual_controller.goalie_process import goalie_process



def main():
    use_sim = True
    vision_q = Queue()
    gc_q = Queue()
    # robot_feedback_q = Queue()

    # inputs
    vision_wkr = Process(target=vision_worker, args=(vision_q, use_sim, 10020,))

    # world model
    wm_manager = WorldModelManager()
    wm_manager.start()
    wm = wm_manager.WorldModel() # pylint: disable=no-member
    wmr = Process(target=wm_runner, args=(wm, vision_q, gc_q,))

    yellow_goalie = Process(target=goalie_process, args=(wm, 'B', "127.0.0.1", 20011, 0, True))
    # blue_goalie = Process(target=goalie_process, args=(wm, 'B', "127.0.0.1", 20011, 0, False))

    # yellow_striker = Process(target=striker_process, args=(wm, 'B', "127.0.0.1", 20011, 1, True))
    blue_striker = Process(target=striker_process, args=(wm, 'B', "127.0.0.1", 20011, 1, False))

    vision_wkr.start()
    wmr.start()
    time.sleep(5)

    # blue_goalie.start()
    yellow_goalie.start()
    # yellow_striker.start()
    blue_striker.start()

    vision_wkr.join()
    wmr.join()
    # yellow_striker.join()
    blue_striker.join()
    yellow_goalie.join()
    # blue_goalie.join()


if __name__ == "__main__":
    main()
