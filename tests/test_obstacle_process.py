def get_scenario_1():
    """
    Scenario 1: Only avoid opponent defense area.
    No other obstacles, just the defense area polygon.

    Returns the target position for the robot to reach.
    """
    # Example target near opponent goal but outside defense area
    main_agent = {'id': 0, 'start': (
        4200, -1550), 'goal': (4100, 1500), 'isYellow': True}

    static_robots = {}  # no static obstacles in this scenario
    dynamic_robots = {}  # no dynamic obstacles either

    return main_agent, static_robots, dynamic_robots


def get_scenario_2():
    """
    Scenario 2: Static opponent as obstacles

    Returns the target position for the robot to reach.
    """
    # Example target near opponent goal but outside defense area
    main_agent = {'id': 0, 'start': (
        0, 0), 'goal': (3000, 0), 'isYellow': True}

    # Static robots is a dictionary of robots (id, start and goal position)
    static_robots = {0: {'start': (1000, 100), 'goal': None, 'isYellow': False},
                     1: {'start': (1000, 0), 'goal': None, 'isYellow': False},
                     2: {'start': (1000, -100), 'goal': None, 'isYellow': False}}
    dynamic_robots = {}  # no dynamic obstacles

    return main_agent, static_robots, dynamic_robots


def get_scenario_3():
    """
    Scenario 3: Static and Dynamic opponent as obstacles

    Returns the target position for the robot to reach.
    """
    # Example target near opponent goal but outside defense area
    main_agent = {'id': 0, 'start': (
        0, -1000), 'goal': (2000, 1200), 'isYellow': True}

    # Static robots is a dictionary of robots (id, start and goal position)
    static_robots = {0: {'start': (800, -300), 'goal': None, 'isYellow': False},
                     1: {'start': (1000, -1000), 'goal': None, 'isYellow': False},
                     2: {'start': (1300, 700), 'goal': None, 'isYellow': False}}

    dynamic_robots = {
        3: {'start': (0, 0), 'goal': (650, -1000), 'isYellow': False}}

    return main_agent, static_robots, dynamic_robots


def get_scenario_4():
    """
    Scenario 4: Static and Dynamic opponent as obstacles

    Returns the target position for the robot to reach.
    """
    # Example target near opponent goal but outside defense area
    main_agent = {'id': 0, 'start': (0, 0), 'goal': (2600, 0),
                  'isYellow': True}

    # Static robots is a dictionary of robots (id, start and goal position)
    static_robots = {0: {'start': (800, -100), 'goal': None, 'isYellow': False},
                     1: {'start': (1600, 100), 'goal': None, 'isYellow': False},
                     2: {'start': (2400, -100), 'goal': None, 'isYellow': False},
                     3: {'start': (2650, -450), 'goal': None, 'isYellow': False},
                     4: {'start': (2550, 350), 'goal': None, 'isYellow': False}}

    dynamic_robots = {}

    return main_agent, static_robots, dynamic_robots


def get_scenario_5():
    """
    Scenario 5: Static and Dynamic opponent as obstacles

    Returns the target position for the robot to reach.
    """
    # Example target near opponent goal but outside defense area
    main_agent = {'id': 0, 'start': (0, 0), 'goal': (3000, 0),
                  'isYellow': True}

    # Static robots is a dictionary of robots (id, start and goal position)
    static_robots = {0: {'start': (800, -100), 'goal': None, 'isYellow': False},
                     1: {'start': (1600, 100), 'goal': None, 'isYellow': False},
                     2: {'start': (2400, -100), 'goal': None, 'isYellow': False}}

    dynamic_robots = {3: {'start': (50, -300), 'goal': (900, 750), 'isYellow': False},
                      4: {'start': (1000, 500), 'goal': (-200, 500), 'isYellow': False}}

    return main_agent, static_robots, dynamic_robots


def get_scenario_6():
    """
    Scenario 6: Static and Dynamic opponent as obstacles

    Returns the target position for the robot to reach.
    """
    # Example target near opponent goal but outside defense area
    main_agent = {'id': 0, 'start': (3700, -1500), 'goal': (3000, 500),
                  'isYellow': True}

    # Static robots is a dictionary of robots (id, start and goal position)
    static_robots = {0: {'start': (3270, -700), 'goal': None, 'isYellow': False},
                     1: {'start': (2750, -700), 'goal': None, 'isYellow': False},
                     2: {'start': (2990, -450), 'goal': None, 'isYellow': False},
                     3: {'start': (2700, 0), 'goal': None, 'isYellow': False}}

    dynamic_robots = {4: {'start': (1200, 0), 'goal': (3500, -1700), 'isYellow': False},
                      5: {'start': (500, -1500), 'goal': (2900, -1300), 'isYellow': False}}

    return main_agent, static_robots, dynamic_robots

