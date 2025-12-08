"""Generic state machine framework."""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, Dict, Type, Any
import time

# Module start time for relative timestamps
_START_TIME = time.time()


TContext = TypeVar("TContext")


class State(ABC, Generic[TContext]):
    """Abstract base class for states in a state machine."""

    @abstractmethod
    def handle_transition(self, context: TContext) -> Optional[Type["State[TContext]"]]:
        """
        Check if state should transition to another state.

        Args:
            context: Context object containing all relevant data

        Returns:
            Next state class if transition should occur, None to stay in current state
        """
        pass

    @abstractmethod
    def execute(self, context: TContext, **kwargs: Any) -> None:
        """
        Execute the behavior for this state.

        Args:
            context: Context object containing all relevant data
            **kwargs: Additional execution-specific parameters
        """
        pass

    def on_enter(self, context: TContext) -> None:
        """
        Called when entering this state.

        Args:
            context: Context object containing all relevant data
        """
        pass

    def on_exit(self, context: TContext) -> None:
        """
        Called when exiting this state.

        Args:
            context: Context object containing all relevant data
        """
        pass


class StateMachine(Generic[TContext]):
    """Generic state machine that manages state transitions."""

    def __init__(self, initial_state: State[TContext]):
        """
        Initialize state machine.

        Args:
            initial_state: The initial state to start in
        """
        self._states: Dict[Type[State[TContext]], State[TContext]] = {}
        self._current_state = initial_state
        self._states[type(initial_state)] = initial_state

    def register_state(self, state: State[TContext]) -> None:
        """
        Register a state with the state machine.

        Args:
            state: State to register
        """
        self._states[type(state)] = state

    def get_current_state_type(self) -> Type[State[TContext]]:
        """Get the type of the current state."""
        return type(self._current_state)

    def get_current_state(self) -> State[TContext]:
        """Get the current state instance."""
        return self._current_state

    def update(self, context: TContext) -> Type[State[TContext]]:
        """
        Update the state machine - check for transitions.

        Args:
            context: Context object containing all relevant data

        Returns:
            Current state type after update
        """
        # Check if current state wants to transition
        next_state_type = self._current_state.handle_transition(context)

        if next_state_type is not None and next_state_type in self._states:
            # Transition to new state
            self._current_state.on_exit(context)
            old_state_name = type(self._current_state).__name__
            self._current_state = self._states[next_state_type]
            self._current_state.on_enter(context)
            new_state_name = type(self._current_state).__name__
            elapsed = time.time() - _START_TIME
            print(f"[STATE @ {elapsed:.3f}s] Forced transition: {old_state_name} -> {new_state_name}")

        return self.get_current_state_type()

    def execute(self, context: TContext, **kwargs: Any) -> None:
        """
        Execute the current state's behavior.

        Args:
            context: Context object containing all relevant data
            **kwargs: Additional execution-specific parameters
        """
        self._current_state.execute(context, **kwargs)

    def force_transition(self, state_type: Type[State[TContext]], context: TContext) -> None:
        """
        Force a transition to a specific state type.

        Args:
            state_type: State type to transition to
            context: Context object for on_exit/on_enter callbacks
        """
        if state_type in self._states and type(self._current_state) is not state_type:
            self._current_state.on_exit(context)
            old_state_name = type(self._current_state).__name__
            self._current_state = self._states[state_type]
            self._current_state.on_enter(context)
            new_state_name = type(self._current_state).__name__
            timestamp = time.time()
            print(
                f"[STATE @ {timestamp:.3f}] Forced transition: {old_state_name} -> {new_state_name}"
            )
            new_state_name = type(self._current_state).__name__
            timestamp = time.time()
            print(
                f"[STATE @ {timestamp:.3f}] Forced transition: {old_state_name} -> {new_state_name}"
            )
