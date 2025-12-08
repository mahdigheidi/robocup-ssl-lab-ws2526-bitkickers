"""Robot architecture module - roles, phases, states, and logging configuration."""

import logging


# ANSI color codes for robot logging
class Colors:
    # Bold colors (for special messages)
    BOLD_GREEN = '\033[1;92m'
    BOLD_CYAN = '\033[1;96m'
    BOLD_YELLOW = '\033[1;93m'
    BOLD_MAGENTA = '\033[1;95m'
    BOLD_BLUE = '\033[1;94m'
    
    # Regular colors (for logging)
    CYAN = '\033[96m'           # Current: PHASE transitions
    YELLOW = '\033[93m'         # Current: state transitions
    GREEN = '\033[92m'          # Alternative option
    MAGENTA = '\033[95m'        # Alternative option
    BLUE = '\033[94m'           # Alternative option
    RED = '\033[91m'            # Alternative option
    
    RESET = '\033[0m'


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    
    def format(self, record):
        # Apply regular (non-bold) colors based on content
        if 'PHASE:' in record.getMessage():
            # Phase transitions in magenta
            record.msg = f"{Colors.MAGENTA}{record.msg}{Colors.RESET}"
        elif 'state:' in record.getMessage():
            # State transitions in cyan
            record.msg = f"{Colors.CYAN}{record.msg}{Colors.RESET}"
        
        return super().format(record)


def enable_logging():
    """Configure logging for robot modules. Call this once at the start of your process."""
    # Configure logging for robot modules ONLY - don't touch root logger
    robot_logger = logging.getLogger('TeamControl.robot')
    robot_logger.setLevel(logging.INFO)
    robot_handler = logging.StreamHandler()
    robot_handler.setFormatter(ColoredFormatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))
    robot_logger.addHandler(robot_handler)
    robot_logger.propagate = False  # Don't propagate to root logger

    state_logger = logging.getLogger('TeamControl.utils.state_machine')
    state_logger.setLevel(logging.DEBUG)
    state_logger.addHandler(robot_handler)
    state_logger.propagate = False  # Don't propagate to root logger
