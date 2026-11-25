"""
main_runner.py

Entry point for running the SSL simulation pipeline.

This script demonstrates how to launch multiple processes for vision processing,
world model management, and robot command generation in parallel. It supports
both simulation (grSim) and real-world setups.

Processes:
    1. vision_worker: Reads vision frames from SSL vision multicast.
    2. wm_runner: Updates the world model using vision and game controller data.
    3. sandbox_process: Sends robot commands to grSim based on the world model.

Queues:
    - vision_q: Transports vision frames from the vision worker to the world model.
    - gc_q: Placeholder queue for game controller events (currently unused).
"""

from multiprocessing import Process, Queue

from team_control.behavior.team import team_control_process
from team_control.vision.vision_worker import vision_worker
from team_control.world.model_manager import WorldModelManager
from team_control.world.model_runner import wm_runner

# in multiprocessing this can only be a simple process


def main():
    """
    Main entry point for launching the SSL multiprocessing pipeline.

    Sets up and runs:
        - Vision worker process to receive frames.
        - World model manager and runner to maintain robot/world state.
        - Sandbox process to send robot commands to grSim.

    Queues:
        vision_q: Queue for vision data.
        gc_q: Queue for game controller data (currently unused).

    Notes:
        - All processes are joined at the end to ensure proper shutdown.
        - This setup is currently configured for simulation (use_sim=True).
        - Ports and other parameters can be adjusted as needed.
    """
    # Simulation parameters
    vision_port = 10020

    # Queues for inter-process communication
    vision_q = Queue()
    gc_q = Queue()  # placeholder for game controller

    # Initialize and start vision worker
    vision_wkr = Process(
        target=vision_worker,
        args=(
            vision_q,
            True,
            vision_port,
        ),
    )

    # Initialize world model managersand
    wm_manager = WorldModelManager()
    wm_manager.start()
    wm = wm_manager.WorldModel()  # type: ignore

    # World model runner process
    wmr = Process(
        target=wm_runner,
        args=(
            wm,
            vision_q,
            gc_q,
        ),
    )

    team = Process(target=team_control_process, args=(wm,))
    # Start all processes
    vision_wkr.start()
    wmr.start()
    team.start()
    # Wait for all processes to finish
    vision_wkr.join()
    wmr.join()
    team.join()


if __name__ == "__main__":
    main()
