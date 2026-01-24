# Robot Architecture

This guide shows you how to implement robot behaviors using our Role → Phase → State architecture.

## Quick Overview

Robot behavior is organized in three levels:

- **Role** (class): What the robot does overall (e.g., `Passer`, `Goalkeeper`)
- **Phase** (enum): Current tactical behavior (e.g., `CHASING`, `PASSING`, `RECEIVING`)
- **State** (automatic): Low-level execution (e.g., `ChasingBallState`, `KickingState`)

**Key idea**: Each role is a self-contained class that defines its phases and maps them to state machines. No global registrations needed.

## File Structure

```
robot/
  context.py             # WorldState, RobotContext, EntityState
  robot.py               # Robot class - executes role behaviors
  role.py                # Role abstract base class
  Movement.py            # Low-level movement controller

  phases/                # Reusable state machine implementations
    ball_handling.py     # Chasing, passing, receiving states + constants

  roles/                 # Role definitions
    passer.py            # Example: two-robot passing drill
```

## Understanding the Components

### WorldState (Single Source of Truth)

Tracks all entities and computes velocities automatically:

```python
world_state = WorldState()

# In your main loop:
world_state.update(wm, timestamp)  # Updates ball and all robot positions/velocities

# Get context for a specific robot:
context = world_state.get_robot_context(robot_id=0, is_yellow=True)
```

### EntityState

Position + velocity for any entity (robot or ball):

```python
@dataclass(frozen=True)
class EntityState:
    position: np.ndarray  # [x, y] for ball, [x, y, orientation] for robots
    velocity: np.ndarray  # [vx, vy] or [vx, vy, vw]
```

### RobotContext

Immutable snapshot of world state for one robot at one point in time:

```python
@dataclass
class RobotContext:
    robot_id: int
    is_yellow: bool
    my_state: EntityState              # My position and velocity
    ball: EntityState                  # Ball position and velocity
    teammates: dict[int, EntityState]  # Other robots on my team (excluding self)
    opponents: dict[int, EntityState]  # All robots on opposing team
    has_possession: bool               # Whether I have the ball
```

**Built automatically from WorldState** - you don't create this manually.

Access data in states:
```python
# Position
my_pos = context.my_state.position          # [x, y, orientation]
ball_pos = context.ball.position            # [x, y]
teammate_pos = context.teammates[1].position  # [x, y, orientation]

# Velocity
my_velocity = context.my_state.velocity     # [vx, vy, vw]
ball_velocity = context.ball.velocity       # [vx, vy]
```

### Role (Abstract Base Class)

All roles extend this:

```python
class Role(ABC):
    @abstractmethod
    def get_initial_phase(self) -> Enum:
Example from `passing_process.py`:
```python
# Create WorldState once
world_state = WorldState()

# In your control loop:
world_state.update(wm, timestamp)  # Update once per frame

if passer_id == 0:
    if check_possession(ball_pos, robot0.position):
        robot0.switch_phase(Passer.Phase.PASSING)  # You control phase
    else:
        robot0.switch_phase(Passer.Phase.CHASING)

# Execute with world_state - context built automatically
robot0.execute_behavior(world_state=world_state, sender=sender, ...)
# States transition automatically inside each phase
```

## Configuration Constants

Behavior tuning constants are defined at the top of state machine files. Example from `phases/ball_handling.py`:

```python
# Distance thresholds (mm)
CHASE_TO_APPROACH_DISTANCE = 200
APPROACH_TO_CHASE_DISTANCE = 250

# Speed and acceleration (mm/s and mm/s²)
CHASE_MAX_SPEED = 5000
CHASE_MAX_ACCELERATION = 1300
APPROACH_MAX_SPEED = 1500
APPROACH_MAX_ACCELERATION = 800

# Passing alignment
ALIGNMENT_THRESHOLD = 0.1  # radians (~5.7 degrees)
KICK_POWER = 5
```

**To tune robot behavior**: Change these constants instead of magic numbers in the code.
- **Phase transitions**: Controlled by your main process (e.g., switch from CHASING to PASSING)
- **State transitions**: Automatic within each phase's state machine

Example from `passing_process.py`:
```python
**2. Use it in your main process:**

```python
from team_control.robot.robot import Robot
from team_control.robot.roles.goalkeeper import Goalkeeper
from team_control.robot.context import WorldState

world_state = WorldState()
goalkeeper = Robot(robot_id=0, is_yellow=True, role=Goalkeeper())
goalkeeper.spawn(x=-4500, y=0, orientation=0)

# In your control loop:
world_state.update(wm, timestamp)  # Update world state

goalkeeper.switch_phase(Goalkeeper.Phase.DIVING)  # Change phases
goalkeeper.execute_behavior(world_state=world_state, sender=sender, ...)
```. Create `roles/goalkeeper.py`:**

```python
from enum import Enum, auto
```python
from team_control.utils.state_machine import State, StateMachine
from team_control.robot.context import RobotContext
from team_control.network.robot_command import RobotCommand
from typing import Optional, Type, Any
import numpy as np

GOAL_CENTER = np.array([-4500, 0])  # Our goal position

class PositioningState(State[RobotContext]):
    """Stay between ball and goal."""

    def handle_transition(self, context: RobotContext) -> Optional[Type[State[RobotContext]]]:
        # State transitions happen automatically based on conditions
        ball_dist = np.linalg.norm(context.ball.position - context.my_state.position[:2])
        if ball_dist < 500:  # Ball too close
            return DivingState
        return None  # Stay in this state

    def execute(self, context: RobotContext, **kwargs: Any) -> None:
        sender = kwargs["sender"]
        movement = kwargs["movement"]  # Passed via execute_behavior

        # Position between ball and goal
        ball_to_goal = GOAL_CENTER - context.ball.position
        target_pos = GOAL_CENTER + 0.3 * ball_to_goal

        vx, vy = movement.go_to(
            context.my_state.position, target_pos, max_speed=2000
        )
        w = movement.face_towards(
            context.my_state.position, context.ball.position
        )

        sender.send_command(
            RobotCommand(context.robot_id, vx, vy, w, kick=0, dribble=0)
        )

class DivingState(State[RobotContext]):
    """Dive to intercept ball."""

    def handle_transition(self, context: RobotContext) -> Optional[Type[State[RobotContext]]]:
        if context.has_possession:
            return PositioningState  # Got the ball, return to position
        return None

    def execute(self, context: RobotContext, **kwargs: Any) -> None:
        sender = kwargs["sender"]
        movement = kwargs["movement"]

        # Aggressive chase
        vx, vy = movement.go_to(
            context.my_state.position, context.ball.position,
            max_acceleration=2000, max_speed=6000
        )
        w = movement.face_towards(context.my_state.position, context.ball.position)
        sender.send_command(
            RobotCommand(context.robot_id, vx, vy, w, kick=0, dribble=1)
        )
class PositioningState(State[RobotContext]):
    """Stay between ball and goal."""

    def handle_transition(self, context: RobotContext):
        # State transitions happen automatically based on conditions
        ball_dist = np.linalg.norm(context.ball_pos - context.my_pos[:2])
        if ball_dist < 500:  # Ball too close
            return DivingState
        return None  # Stay in this state

    def execute(self, context: RobotContext, **kwargs):
        sender = kwargs["sender"]

        # Position between ball and goal
        ball_to_goal = GOAL_CENTER - context.ball_pos
        target_pos = GOAL_CENTER + 0.3 * ball_to_goal

        vx, vy = context.my_movement.go_to(
            context.my_pos, target_pos, max_speed=2000
        )
        w = context.my_movement.face_towards(
            context.my_pos, context.ball_pos
        )

        sender.send_command(
            RobotCommand(context.robot_id, vx, vy, w, kick=0, dribble=0)
        )

class DivingState(State[RobotContext]):
    """Dive to intercept ball."""

    def handle_transition(self, context: RobotContext):
        if context.has_possession:
            return PositioningState  # Got the ball, return to position
        return None

    def execute(self, context: RobotContext, **kwargs):
        sender = kwargs["sender"]
        # Aggressive chase
        vx, vy = context.my_movement.go_to(
            context.my_pos, context.ball_pos,
            max_acceleration=2000, max_speed=6000
        )
        w = context.my_movement.face_towards(context.my_pos, context.ball_pos)
        sender.send_command(
            RobotCommand(context.robot_id, vx, vy, w, kick=0, dribble=1)
        )

class PositioningStateMachine(StateMachine[RobotContext]):
    """State machine for goalkeeper positioning."""

    def __init__(self):
        positioning = PositioningState()
        diving = DivingState()

        super().__init__(initial_state=positioning)
        self.register_state(diving)
```

**2. Export it in `phases/__init__.py`:**

```python
from team_control.robot.phases.goalkeeping import PositioningStateMachine

__all__ = [
    "ChasingBallStateMachine",
    "PassingStateMachine",
    "ReceivingStateMachine",
    "PositioningStateMachine",  # Add your new one
]
```

**3. Use it in your role** (already shown above in `Goalkeeper.create_phase_machine`)

## Reusing Existing State Machines

Check `phases/ball_handling.py` for ready-to-use state machines:

- `ChasingBallStateMachine` - Chase and capture ball
- `PassingStateMachine` - Align and kick to target
- `ReceivingStateMachine` - Position to intercept passes

Example - both roles can clear the ball the same way:

```python
class Attacker(Role):
    def create_phase_machine(self, phase):
        if phase == self.Phase.CLEARING:
            return PassingStateMachine()  # Reuse!

class Goalkeeper(Role):
    def create_phase_machine(self, phase):
        if phase == self.Phase.CLEARING:
            return PassingStateMachine()  # Same!
```

## Debugging

Phase and state transitions are automatically logged:

```
[02:31:09] R0[Passer] PHASE: RECEIVING -> CHASING
    R0 state: ChasingBallState -> ApproachingBallState
```

- **Magenta**: Phase transitions (tactical changes)
- **Cyan**: State transitions (execution changes)

Access current state for debugging:
```python
print(robot.current_state_name)  # "ChasingBallState"
print(robot.current_phase)        # Passer.Phase.CHASING
```

## Best Practices

✅ **DO**: Reuse state machines from `phases/` when possible
✅ **DO**: Keep role-specific logic in the role file
✅ **DO**: Control phase transitions in your main process
✅ **DO**: Let states transition automatically

❌ **DON'T**: Add role-specific logic to shared state machines
❌ **DON'T**: Manually trigger state transitions

---

**Next**: See `roles/passer.py` and `ssl/manual_controller/passing_process.py` for a complete working example.

