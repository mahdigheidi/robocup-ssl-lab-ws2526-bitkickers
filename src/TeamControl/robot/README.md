# Robot Architecture

This document explains the robot behavior system architecture. Read this before adding new robot behaviors or modifying existing ones.

## Overview

The robot system uses a three-level hierarchy: **Role → Phase → State**

- **Role**: What position the robot plays (e.g., Attacker, Goalkeeper, PassingDemo)
- **Phase**: What the robot is currently doing (e.g., Chasing, Passing, Receiving)
- **State**: Low-level execution details within a phase (e.g., ChasingBallState, KickingState)

**Key principle**: Roles are types (classes), Phases are runtime enums, States are state machine implementations.

## Why This Design?

**Problem**: The original architecture used string-based roles ("passer", "receiver") and required hard-coded dictionaries mapping roles to states. Adding new behaviors meant updating multiple registration points.

**Solution**: 
- Each role is a class that defines its own valid phases
- Phases map to reusable state machine implementations
- No global registration - each role self-describes its capabilities
- Type safety prevents invalid role/phase combinations

**Benefit**: Adding a new role or behavior is localized to one file. No hunting for registration dictionaries.

## Directory Structure

```
robot/
  README.md              # This file
  context.py             # RobotContext - shared data for state machines
  robot.py               # Robot class - generic over role type
  Movement.py            # Low-level movement controller
  spawn_robot.py         # Simulator integration
  
  phases/                # Shared, reusable state machine implementations
    __init__.py
    ball_handling.py     # Chasing, Passing, Receiving, Dribbling
    shooting.py          # Shooting state machines
## File Structure

```
robot/
  context.py             # RobotContext dataclass - shared data for state machines
  robot.py               # Robot class - manages role instance and phase transitions
  role.py                # Role abstract base class
  Movement.py            # Low-level movement controller
  spawn_robot.py         # Simulator spawn commands
  
  phases/                # Reusable state machine implementations
## Core Components

### RobotContext (`context.py`)

Dataclass containing all information needed for decision-making:

```python
@dataclass(slots=True)
class RobotContext:
    robot_id: int
    my_pos: np.ndarray           # [x, y, orientation]
    my_movement: RobotMovement   # Movement controller instance
    other_pos: np.ndarray        # Other robot position
    ball_pos: np.ndarray         # Ball position [x, y]
    ball_velocity: np.ndarray    # Ball velocity [vx, vy]
    has_possession: bool         # Whether robot has ball
```

**Important**: This is passed to every state's `execute()` method. Add fields here if all states need new information.

### Role (`role.py`)

Abstract base class that all roles must extend:

```python
class Role(ABC):
    @abstractmethod
    def get_initial_phase(self) -> Enum:
        """Return starting phase when robot spawns."""
        pass
    
    @abstractmethod
    def create_phase_machine(self, phase: Enum) -> StateMachine[RobotContext]:
        """Create state machine for the given phase."""
        pass
```

**Your role must**:
1. Define a nested `Phase` enum
2. Implement `get_initial_phase()` to return a Phase value
3. Implement `create_phase_machine()` to instantiate the right state machine for each phase

### Robot (`robot.py`)

Generic class `Robot[R]` where R is the role type:

## How To: Add a New Role

**Example**: Create a `Goalkeeper` role

1. **Create `roles/goalkeeper.py`**:

```python
from enum import Enum, auto
from TeamControl.robot.role import Role
from TeamControl.robot.context import RobotContext
from TeamControl.utils.state_machine import StateMachine
from TeamControl.robot.phases import ChasingBallStateMachine, PassingStateMachine

class Goalkeeper(Role):
    """Goalkeeper role - defends goal."""
    
    class Phase(Enum):
        POSITIONING = auto()  # Position in goal
        DIVING = auto()       # Dive to block shot
        CLEARING = auto()     # Clear ball from goal area
    
    def get_initial_phase(self) -> Enum:
        return self.Phase.POSITIONING
    
    def create_phase_machine(self, phase: Enum) -> StateMachine[RobotContext]:
        if phase == self.Phase.POSITIONING:
            return PositioningStateMachine()  # You'll create this
        elif phase == self.Phase.DIVING:
            return DivingStateMachine()       # You'll create this
        elif phase == self.Phase.CLEARING:
            return PassingStateMachine()      # Reuse existing!
        else:
            raise ValueError(f"Unknown phase: {phase}")
## How To: Add a New Shared Capability

**Example**: Create a `ShootingStateMachine`

1. **Create state classes in `phases/shooting.py`**:

```python
from TeamControl.utils.state_machine import State, StateMachine
from TeamControl.robot.context import RobotContext

class AimingState(State[RobotContext]):
    def handle_transition(self, context: RobotContext):
        if aligned_with_goal(context):
            return ShootingState
        return None
    
    def execute(self, context: RobotContext, **kwargs):
        sender = kwargs["sender"]
        # Aim toward goal
        w = context.my_movement.face_towards(context.my_pos, GOAL_POS, 3)
        sender.send_command(RobotCommand(context.robot_id, 0, 0, w, 0, 1))

class ShootingState(State[RobotContext]):
    # ... implementation

class ShootingStateMachine(StateMachine[RobotContext]):
    def __init__(self):
        aiming = AimingState()
        shooting = ShootingState()
        super().__init__(initial_state=aiming)
        self.register_state(shooting)
```

2. **Export in `phases/__init__.py`**:

```python
from TeamControl.robot.phases.shooting import ShootingStateMachine

__all__ = ["ChasingBallStateMachine", "PassingStateMachine", 
           "ReceivingStateMachine", "ShootingStateMachine"]
## Understanding Phase Transitions

**Phase transitions are controlled by your main process**, not by the robot itself.

Example from `passing_process.py`:

```python
# Determine which robot should chase the ball
passer_id = determine_ball_responsibility(robot0.position, robot1.position, ball_pos, ball_velocity)

if passer_id == 0:
    if check_possession(ball_pos, robot0.position):
        robot0.switch_phase(PassingDemo.Phase.PASSING)  # Has ball → pass
    else:
        robot0.switch_phase(PassingDemo.Phase.CHASING)  # No ball → chase
    robot1.switch_phase(PassingDemo.Phase.RECEIVING)     # Other robot receives
else:
    # Reverse roles
    robot1.switch_phase(...)
    robot0.switch_phase(PassingDemo.Phase.RECEIVING)
```

**State transitions are automatic** within a phase's state machine. You don't call them manually.

## Debugging and Logging

Phase and state transitions are automatically logged:

```
[02:31:09] R0[PassingDemo] PHASE: RECEIVING -> CHASING
    R0 state: ChasingBallState -> ApproachingBallState
[02:31:10] R0[PassingDemo] PHASE: CHASING -> PASSING
    R0 state: AligningWithTargetState -> KickingState
```

- **Magenta**: Phase transitions (high-level strategy changes)
- **Cyan**: State transitions (low-level execution changes)

Logging is configured in the main process (e.g., `passing_process.py`). See that file for the logging setup if you need to adjust it.

## Common Patterns

### Sharing State Machines Between Roles

Good! That's the point of `phases/`:

```python
# Both Attacker and Goalkeeper can clear the ball
class Attacker(Role):
    def create_phase_machine(self, phase):
        if phase == self.Phase.CLEARING:
            return PassingStateMachine()  # Reuse!

class Goalkeeper(Role):
    def create_phase_machine(self, phase):
        if phase == self.Phase.CLEARING:
            return PassingStateMachine()  # Same implementation!
```

### Role-Specific Behavior

Keep it in the role file, not in shared phases:

```python
# DON'T add role-specific logic to PassingStateMachine
# DO create a new state machine in your role file if needed

class Goalkeeper(Role):
    # ... in roles/goalkeeper.py, not phases/
```

### Accessing Current State

For debugging or conditional logic:

```python
current_state_name = robot.current_state_name  # Returns e.g., "ChasingBallState"
current_phase = robot.current_phase  # Returns Phase enum value
```
1. Create `roles/new_role.py`
2. Define role class extending `Role`
3. Define role-specific `Phase` enum (nested class)
4. Implement `get_initial_phase()`, `get_valid_phases()`, `create_phase_machine()`
5. Compose existing phase state machines or create new ones

### Adding a New Shared Capability

1. Create state machine in `phases/` directory
2. Implement with `State[RobotContext]` interface
3. Import and use in any role's `create_phase_machine()`

### Adding Role-Specific Behavior

Keep it in the role file - don't pollute shared phases with role-specific logic.

## Terminology Reference

| Term | Level | Nature | Example |
|------|-------|--------|---------|
| **Role** | Strategic | Type (class) | Attacker, Goalkeeper |
| **Phase** | Tactical | Runtime (enum) | Chasing, Passing, Diving |
| **State** | Execution | Runtime | Accelerating, Aligning, Kicking |
| **Capability** | - | Shared implementation | PassingStateMachine |

## Philosophy

The architecture balances:
- **Flexibility**: Easy to add roles and capabilities
- **Type safety**: Prevent invalid configurations at compile time
- **Reusability**: Share code across roles
- **Clarity**: Clear separation of strategic vs tactical vs execution logic
- **Simplicity**: Avoid over-engineering, prefer explicit over magic

Each robot is autonomous, with its own role and phase state machine. The system coordinates robots through shared world state (RobotContext) rather than direct communication.

---

*This architecture is designed to scale from simple two-robot drills to full 11v11 soccer with complex formations and tactics.*
