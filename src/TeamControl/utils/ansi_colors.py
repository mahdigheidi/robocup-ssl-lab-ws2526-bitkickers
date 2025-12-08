from enum import Enum


class AnsiColors(Enum):
    """ANSI color codes for logging"""

    # Bold colors (for special messages)
    BOLD_GREEN = "\033[1;92m"
    BOLD_CYAN = "\033[1;96m"
    BOLD_YELLOW = "\033[1;93m"
    BOLD_MAGENTA = "\033[1;95m"
    BOLD_BLUE = "\033[1;94m"

    # Regular colors (for logging)
    CYAN = "\033[96m"  # Current: PHASE transitions
    YELLOW = "\033[93m"  # Current: state transitions
    GREEN = "\033[92m"  # Alternative option
    MAGENTA = "\033[95m"  # Alternative option
    BLUE = "\033[94m"  # Alternative option
    RED = "\033[91m"  # Alternative option

    RESET = "\033[0m"
