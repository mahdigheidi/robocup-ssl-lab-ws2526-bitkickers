"""Robot architecture module - roles, phases, states, and logging configuration."""

import logging

from team_control.utils.ansi_colors import AnsiColors


def enable_logging():
    class ColoredFormatter(logging.Formatter):
        """Custom formatter that adds colors to log messages."""

        def format(self, record):
            # Apply regular (non-bold) colors based on content
            if "PHASE:" in record.getMessage():
                # Phase transitions in magenta
                record.msg = f"{AnsiColors.MAGENTA}{record.msg}{AnsiColors.RESET}"
            elif "state:" in record.getMessage():
                # State transitions in cyan
                record.msg = f"{AnsiColors.CYAN}{record.msg}{AnsiColors.RESET}"

            return super().format(record)

    """Configure logging for robot modules. Call this once at the start of your process."""
    # Configure logging for robot modules ONLY - don't touch root logger
    robot_logger = logging.getLogger("team_control.behavior")
    robot_logger.setLevel(logging.INFO)
    robot_handler = logging.StreamHandler()
    robot_handler.setFormatter(ColoredFormatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    robot_logger.addHandler(robot_handler)
    robot_logger.propagate = False  # Don't propagate to root logger

    state_logger = logging.getLogger("team_control.utils.state_machine")
    state_logger.setLevel(logging.DEBUG)
    state_logger.addHandler(robot_handler)
    state_logger.propagate = False  # Don't propagate to root logger
