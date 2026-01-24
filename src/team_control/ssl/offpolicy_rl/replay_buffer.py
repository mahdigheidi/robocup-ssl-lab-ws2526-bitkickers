from collections import deque
import random

import torch


class ReplayBuffer:
    """
    Transition-based replay buffer that supports structured states:
    state = (robot, ball, teammates, opponents)
    stores (states, actions, rewards, next_states, dones)
    """

    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self,
        robot, ball, teammates, opponents, # State
        action, # Action
        reward, # Reward
        next_robots, next_ball, next_teammates, next_opponents, # Next state
        done # Done
    ):
        self.buffer.append((
            robot, ball, teammates, opponents,
            action, reward,
            next_robots, next_ball, next_teammates, next_opponents,
            done,
        ))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        """
        Returns:
            states, actions, rewards, next_states, dones
        where states and next_states are lists of structured states.
        """
        batch = random.sample(self.buffer, batch_size)

        batch_robot, bathc_ball, batch_teammates, batch_opponents, batch_actions, batch_rewards, batch_next_robot, batch_next_ball, batch_next_teammate, batch_next_opponent, batch_done = zip(*batch)

        actions = torch.stack(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        return states, actions, rewards, next_states, dones


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)

    print("Running ReplayBuffer tests...\n")

    buffer = ReplayBuffer(capacity=100)

    def make_state(B=1, n_team=None, n_opp=None):
        """
        Creates a dummy structured state:
        (robot, ball, teammates, opponents)
        """
        robot = torch.randn(B, 9)
        ball = torch.randn(B, 4)

        if n_team is None:
            team = None
        else:
            team = torch.randn(n_team, 9)

        if n_opp is None:
            opp = None
        else:
            opp = torch.randn(n_opp, 9)

        return (robot, ball, team, opp)

    # -------------------------
    # Test 1: Basic push & length
    # -------------------------
    print("Test 1: push() and __len__()")
    s = make_state()
    a = torch.randn(5)
    r = 1.0
    s2 = make_state()
    d = False

    buffer.push(s, a, r, s2, d)
    assert len(buffer) == 1, "Buffer length should be 1 after one push"
    print("  ✓ Passed\n")

    # -------------------------
    # Test 2: Multiple pushes
    # -------------------------
    print("Test 2: multiple transitions")
    for _ in range(9):
        buffer.push(make_state(), torch.randn(5), random.random(), make_state(), False)

    assert len(buffer) == 10, f"Buffer length should be 10, got {len(buffer)}"
    print("  ✓ Passed\n")

    # -------------------------
    # Test 3: Sampling shapes and types
    # -------------------------
    print("Test 3: sample() return structure")
    batch_size = 4
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    assert isinstance(states, tuple), "states must be a tuple"
    assert isinstance(next_states, tuple), "next_states must be a tuple"
    assert actions.shape == (batch_size, 5), f"actions shape mismatch: {actions.shape}"
    assert rewards.shape == (batch_size, 1), f"rewards shape mismatch: {rewards.shape}"
    assert dones.shape == (batch_size, 1), f"dones shape mismatch: {dones.shape}"

    print("  ✓ states and next_states are tuples")
    print("  ✓ actions shape:", actions.shape)
    print("  ✓ rewards shape:", rewards.shape)
    print("  ✓ dones shape:", dones.shape)
    print()

    # -------------------------
    # Test 4: Structured state integrity
    # -------------------------
    print("Test 4: structured state integrity")
    sample_state = states[0]
    assert isinstance(sample_state, tuple) and len(sample_state) == 4, \
        "Each state must be a tuple: (robot, ball, teammates, opponents)"

    robot, ball, team, opp = sample_state
    assert isinstance(robot, torch.Tensor) and robot.shape[-1] == 9
    assert isinstance(ball, torch.Tensor) and ball.shape[-1] == 4
    if team is not None:
        assert isinstance(team, torch.Tensor) and team.shape[-1] == 9
    if opp is not None:
        assert isinstance(opp, torch.Tensor) and opp.shape[-1] == 9

    print("  ✓ State structure preserved")
    print()

    # -------------------------
    # Test 5: Variable-sized teammates/opponents
    # -------------------------
    print("Test 5: variable-sized teammates/opponents")
    buffer = ReplayBuffer(capacity=50)

    for _ in range(10):
        s = make_state(n_team=random.randint(0, 4), n_opp=random.randint(0, 4))
        a = torch.randn(5)
        r = random.random()
        s2 = make_state(n_team=random.randint(0, 4), n_opp=random.randint(0, 4))
        d = random.choice([True, False])
        buffer.push(s, a, r, s2, d)

    states, actions, rewards, next_states, dones = buffer.sample(5)

    for st in states:
        _, _, team, opp = st
        if team is not None:
            assert team.ndim == 2 and team.shape[1] == 9
        if opp is not None:
            assert opp.ndim == 2 and opp.shape[1] == 9

    print("  ✓ Handles variable numbers of agents correctly")
    print()

    # -------------------------
    # Test 6: Capacity overflow
    # -------------------------
    print("Test 6: capacity overflow behavior")
    buffer = ReplayBuffer(capacity=5)

    for i in range(10):
        buffer.push(make_state(), torch.randn(5), i, make_state(), False)

    assert len(buffer) == 5, f"Buffer should cap at 5, got {len(buffer)}"
    print("  ✓ Old transitions discarded when capacity exceeded")
    print()

    print("All ReplayBuffer tests passed successfully.")