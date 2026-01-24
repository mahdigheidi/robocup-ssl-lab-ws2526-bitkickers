from team_control.ssl.rl.reward import Reward

# -------------------------------------------------------
# Simple test scenarios for Reward
# -------------------------------------------------------

class DummyRobot:
    def __init__(self, x, y, o=0.0):
        self.position = (x, y, o)


class DummyBall:
    def __init__(self, x, y):
        self.position = (x, y)


class DummyField:
    """
    Minimal stand‑in for Field so we can test reward logic
    without running the simulator.
    """

    def __init__(self):
        self.x_max = 4500
        self.y_max = 3000
        self.x_min = -4500
        self.y_min = -3000
        self.goal_width = 1000
        self.penalty_width = 2000
        self.penalty_depth = 1000
        self.div_name = "B"


def print_case(title, reward, done):
    print("=" * 60)
    print(title)
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print("=" * 60)


def test_field_player_yellow():
    print("\nTEST: Yellow Field Player")

    field = DummyField()
    reward_fn = Reward(role=1, debug_on=True)

    # Robot close to ball, facing it
    robot = DummyRobot(x=-1000, y=0, o=0)
    ball = DummyBall(x=-900, y=0)

    teammates = []
    opponents = []

    reward, done = reward_fn.calculate_reward(
        field, True, robot, ball, teammates, opponents
    )
    print_case("Yellow Field Player: Near ball, facing it", reward, done)


def test_field_player_blue():
    print("\nTEST: Blue Field Player")

    field = DummyField()
    reward_fn = Reward(role=1, debug_on=True)

    # Robot far from ball
    robot = DummyRobot(x=3000, y=2000, o=0)
    ball = DummyBall(x=-3000, y=-2000)

    reward, done = reward_fn.calculate_reward(
        field, False, robot, ball, [], []
    )
    print_case("Blue Field Player: Far from ball", reward, done)


def test_goalie_yellow():
    print("\nTEST: Yellow Goalie")

    field = DummyField()
    reward_fn = Reward(role=0, debug_on=True)

    # Goalie inside defense, ball also inside defense
    robot = DummyRobot(x=4300, y=0, o=0)
    ball = DummyBall(x=4200, y=0)

    reward, done = reward_fn.calculate_reward(
        field, True, robot, ball, [], []
    )
    print_case("Yellow Goalie: Ball inside defense area (save)", reward, done)


def test_goalie_blue():
    print("\nTEST: Blue Goalie")

    field = DummyField()
    reward_fn = Reward(role=0, debug_on=True)

    # Ball inside goal line -> conceded
    robot = DummyRobot(x=-4300, y=0, o=0)
    ball = DummyBall(x=-4600, y=0)

    reward, done = reward_fn.calculate_reward(
        field, False, robot, ball, [], []
    )
    print_case("Blue Goalie: Goal conceded", reward, done)


def test_collision_field_player():
    print("\nTEST: Field Player Collision")

    field = DummyField()
    reward_fn = Reward(role=1, debug_on=True)

    robot = DummyRobot(x=0, y=0, o=0)
    teammate = DummyRobot(x=10, y=10, o=0)  # close enough to collide
    ball = DummyBall(x=1000, y=0)

    reward, done = reward_fn.calculate_reward(
        field, True, robot, ball, [teammate], []
    )
    print_case("Collision with teammate", reward, done)


if __name__ == "__main__":
    print("\nRunning Reward class unit tests...")

    test_field_player_yellow()
    test_field_player_blue()
    test_goalie_yellow()
    test_goalie_blue()
    test_collision_field_player()

    print("\nAll Reward tests executed.")
