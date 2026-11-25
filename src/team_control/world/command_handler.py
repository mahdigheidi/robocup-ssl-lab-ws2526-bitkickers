from enum import Enum

from team_control.game_controller.common import Command, Stage


class STATE(Enum):
    RUNNING = 0
    HALTED = 1
    STOPPED = 2
    TIMEOUT = 3

    def update_stage(self,stage):
        print(stage)
        if stage in (Stage.NORMAL_FIRST_HALF_PRE, Stage.EXTRA_FIRST_HALF_PRE):
            return 
        elif stage in (Stage.NORMAL_SECOND_HALF_PRE, Stage.EXTRA_SECOND_HALF_PRE):
            return 
        
        # first, second half
        elif stage in (Stage.NORMAL_FIRST_HALF, Stage.EXTRA_FIRST_HALF):
            return 
        elif stage in (Stage.NORMAL_SECOND_HALF, Stage.EXTRA_SECOND_HALF):
            return 
        
        # half time breaks
        elif stage == Stage.NORMAL_HALF_TIME:
            return 
        elif stage == Stage.EXTRA_TIME_BREAK:
            return 
        elif stage == Stage.EXTRA_HALF_TIME:
            return
        
        # Penalty 
        elif stage == Stage.PENALTY_SHOOTOUT_BREAK:
            return 
        elif stage == Stage.PENALTY_SHOOTOUT:
            return 
        
        # post game
        elif stage == Stage.POST_GAME:
            return 

    
    def update_state(self,command):
        print(command) 
        if command == Command.HALT:
            return STATE.HALTED
        elif command == Command.STOP:
            return STATE.STOPPED
        elif command in (Command.NORMAL_START, Command.FORCE_START):
            return STATE.RUNNING
        
        #timeout
        elif command == Command.TIMEOUT_BLUE:
            return STATE.TIMEOUT
        elif command == Command.TIMEOUT_YELLOW:
            return STATE.TIMEOUT
        
        elif command == Command.PREPARE_KICKOFF_BLUE:
            return
        elif command == Command.PREPARE_KICKOFF_YELLOW:
            return
        
        elif command in (Command.DIRECT_FREE_BLUE, Command.INDIRECT_FREE_BLUE):
            return
        elif command in (Command.DIRECT_FREE_YELLOW, Command.INDIRECT_FREE_YELLOW):
            return

        # Ball placements
        elif command == Command.BALL_PLACEMENT_BLUE:
            return
        elif command == Command.BALL_PLACEMENT_YELLOW:
            return
        
        # Penalty
        elif command == Command.PREPARE_PENALTY_YELLOW:
            return
        elif command == Command.PREPARE_PENALTY_BLUE:
            return
            