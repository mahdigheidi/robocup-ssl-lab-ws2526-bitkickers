I will update the documentation here soon...

# Match length
Final match will run for 6 minutes, we have set the grsim to have 60fps and the update interval of the world is 5 so every 1/12 second we receive new data, then the episode time length = episode length / 12 for example if the steps in every episode is 512 then the episode will take ~45s and running the code for 2000 episodes will take 24 hours of wall clock time.

# Motivation
AlphaGo, AlphaZero, OpenAI Five

## Plasiticity loss and the twohot encoding idea

# Design documentation
We set the thickness of the walls in all enviornments to zero, so agents can go whereever they want, but at the same time, we punish them for being out of the field and away from the ball.

Without supervised learning in the beginning based on what we have done in the goalie, striker, passing interceptor and mover phases we have to let the robots self-play to learn the physics of the model and the basic skills, so our curriculum learning would take a long time, but with supervised learning with datasets created from our data it already knows some of the skills, like moving, rotating, kicking and dribbling. which puts us ahead hugely. then only using self-play and reinforcement learning it has to learn first, adverserial play with opponents, then we add other players so that they can learn cooperative team-play and more strategic behavior. This reduces our curriculum hugely. If we haven't had used this approach, we had to first let a single robot roam around in a small field, with very sparse rewards, to get to the ball, to kick the ball, to learn to rotate and dribble and ....... and then when it learned this playing style we needed to add a goalie and then let them play together. Another drawback is the training time. Since we don't have an environment that we can get the next states deterministicly we have to get the frames and the data from grSim. that introduces a really heavy bottleneck which increases the training time orders of magnitude.
Now, we train our actor model first using the data generated from the previous behaviours that we have already. For the goalie agent, we have the goalie process logic that predicts the ball trajectory and moves towards the point that its coming. We store this data in a csv file and train our goalie agent on it. Then we have the striker process already implemented which is a combination of the mover process and the shooting process. There are also scenarios that we have coded manually to generate the data so that the agent can use to learn.
We train the actor model first on this data. Then we observe the behavior of the model and make sure everything is working perfectly using visualization on grSim. After that we move on to the RL self-play style. We add the agents and let them play together using the rewards that we have already defined.

This design choice but also made us face a difficulty, if we didn't use the supervised learning we could've included an LSTM so that the model kept a memory of previous states and frames by itself and just decide on the previous states. but now that we are using

off-policy vs on-policy?
We **can't** use off-policy since we are doing self-play. Because suppose that we're using off-policy and we have a replay buffer. Replay buffer has actions from previous episodes from which we have diverged, so we're basically updating on trash data. We have to be on-policy. The difficulty with on-policy methods is that they're starting off

Think of off-policy as:

“Learn a value function for a fixed world.”

Think of on-policy as:

“Learn a strategy in a changing world.”
Physics problems:

“The world does not care that I am learning.”

Strategy problems:

“The world changes because I am learning.”
Type of world
Best algorithm family
World is a machine (fixed dynamics)
Off-policy (TD3, SAC)
World is an opponent (adaptive)
On-policy (PPO, MAPPO


but for PPO there is a cold start problem, we remove this by training it on our models.

## GrSim:
- Running and getting grSim to work was an absolute pain on MAC. both because of the old resources that the grSim uses(such as old cmake, ode, etc). and also because of the macos underlying network structure and how it handles the multicasting. We have tried to explain our approach here in the following so people can use this data to use it to run it on mac in the most straightforward way possible.
- Here we mentions the issues and the ways we have solved them. Also we mention the lessons learned.

grSim installation on Mac Arm Machines(M1 onwards)

**Very important note**: grSim will be compiled with CMake < 4, if you have CMake >= 4 you have to download v3.X.X matching your arch from this link[https://github.com/Kitware/CMake/releases].

--> install CMake 3.X.X
brew install pkg-config
brew install ode
brew install qt@5
**Note**: as grSim uses a very old version of protobuf/vartypes and brew doesn't have it, skip the installation so the CMake downloads and builds the required version.

git clone git@github.com:RoboCup-SSL/grSim.git
cd grSim/
mkdir build
cd build

./path/to/cmake3.X.X -DCMAKE_INSTALL_PREFIX=/opt/homebrew/Cellar ..
**Note**: in the grSim offical install instructions that is denoted as /usr/local which is the path on intel based machines, for installation with homebrew use /opt/homebrew/Cellar


After you install it cd ../bin/ and ./grSim to run it. (or if you want just cp it to /opt/homebrew/bin so its visible from the PATH)

**Issue**: you may see the following message repeatedly in the console log "Sending UDP datagram failed (maybe too large?). Size was: X byte(s)."
this issue is related to how MacOS handles packet sending on multicast mode, to resolve this issue set the 'Vision multicast address' to the broadcast address of your current network interface, for example if you're using wifi and network interface is "en0" just ifconfig -> look for en0 -> get the inet broadcast address and set it for vision multicast address.
You also have to configure the client to use this for vision worker.

# Ubuntu/VM
sudo apt install git build-essential cmake pkg-config qtbase5-dev libqt5opengl5-dev libgl1-mesa-dev libglu1-mesa-dev libprotobuf-dev protobuf-compiler libode-dev libboost-dev
sudo apt install tini libqt5opengl5 libode8 x11vnc xvfb
cd grSim/
mkdir build
cd build/
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
make
cd ../..

# run grSim headless
grSim -H -platform offscreen
# grsim runs with division B by default

# MacOS ARM Series


---------------------

## Technical implementation:
In this project I have tried to stay true to the best software engineering practices to the extent of my knowledge. Although I may have not 100 percent succeeded but it's something.
Main ideas our project is built on
- gpu usage doesn't make sense here in this project, because of the multiprocess nature of this setting
### Curriculum learning
Curriculum will tell us in which field and with how many players and with how much skill range play
### Imitation learning
We use the strategy and the data of our prior implementations to have a baseline for our agents/
### Deep Reinforcement learning
Then we improve them using self-play
Mention that we have tried PPO, but it didn't work because of its off-policy nature. PPO TD3 SAC HER
give a history of what we tried to do, then polish it with the final implementation.
### Two hot encoding

### Action masking
Using curriculum learning we implement action masking to enable the agent to learn the skills required to play the game and then build on that.

## How to take care of the sparse reward settings
- We can introduce dense rewards in our environment (distance to the ball, goals, facing the ball, collisions, ...) -> we chose this one
- Methods like HER(hindsight experience replay), curiousity driven
- our goal is not always straightforward so we cannot use HER method directly, curiousity driven RL also need at least 3 more networks and since our settings is pretty straightforward the dense reward setting that we created is enough


### papers i have read

### Encoder
Both of the models will be using the same encoder, hence we train the encoder on a lot of data, freeze and then store it and give it to the models.

### Explain design choices

## Assumptions:
Our assumptions is that we're always playing on the right side of the field. If we are supposed to play on the left side, then we mathematically change the model inputs. Our variable is_team_yellow throughout this project can be equally interpreted as if the team is playing on the right side.
Our model has the assumption that the robot belongs to the right side of the field and plays on the right-side. Hence, if the input data belongs to left side players we transform the info to the projection of the right side, and then transform the commands back to the left side.

## multiple grSim environments
Since we are using curriculum learning, and we separate our player fields and our goalies, we have created multiple grSim environment and trained these agents simultaneosly. For example the goalie and the field player are being trained at the same time using two different grSim envs.

## training on VMs
- run grSim headless so that its okay

<!-- # RoboCup SSL - Robot Control

This is the code you will need to control the SSL robots in grSim and also to control the real robots.

## Installation

To install this module, do

```
git clone https://github.com/WSU-TurtleRabbit/2025-teamcontrol.git
```

### Create the virual environment and activate it
Then we will have to create a virtual environment.
In Linux :
```bash
python3 -m venv .venv
```
In Windows :
```powershell
python -m venv .venv
```
*if this did not work, it is recommended to get the version of python (>=3.13) from the [python.org](https://www.python.org/downloads/). Then add python to path
in particular : Appdata/Python3.13/ (-> python) and Appdata/Python3.13/Scripts/ (-> pip)
Then restart vscode or the app with terminal opened.


Then we will have to install the pip packages and our development project.
To do so, you will first have to activate the environment.
In Linux :
```bash
source .venv/bin/activate
```
In Windows :
```powershell
.\.venv\Scripts\activate.bat
```

### Install necessary modules
Then install the modules and this project as *Editable* project (that uses `pyproject.toml`)
```bash
pip install -e .
```

Now you should be able to run and start coding without any problem !

If you want to know what was installed (and what version), you can do :
```bash
pip list
```

### Deactivate the virtual environment
In Linux :
```bash
deactivate
```
In Windows :
```powershell
.\.venv\Scripts\deactivate.bat
```

## Repository structure

- **docs/** – Contains documentation, and usage guides.
- **src/** – Houses the core source code for the project.
  - **ball/** - Contains modules for ball spawning.
  - **network/** – Handles communication between the PC and the robots/simulation.
  - **robot/** – Contains modules for robot behavior.
  - **ssl/** – Implements Small Size League–specific software (game controller, grSim, vision).
  - **utils/** – General-purpose helper functions.
  - **world/** – Everything related to the world model.
  - **main.py** – Primary entry script to start the system (on the real robots).
- **tests/** – Includes test suites to ensure code reliability and correctness.
- **pyproject.toml** – Defines project metadata, dependencies, and build configuration.
- **setup.sh** – Shell script to set up the development environment.
- **README.md** – Main documentation file (this file).

## Code

Most of the code in this repository is provided as a framework, so you do not need to modify everything. To help you get started, here are the main files and functions that are relevant for your lab work:

### Key Functions & Entry Points
- **Movement Control**
    - `goToPoint` function
    - Location: `src/robot/movement.py`
    - This is the function you should improve for the first assignment.
    - Feel free to add more functions to `movement.py`
- **Simulation Entry Point**
    - Start the grSim simulation
    - Location: `src/ssl/grSim/sandbox.py`
    - This is where the simulation begins. You can run this to see your robot behavior in action.
- **Behavior Setup**
    - This is the process `sandbox.py` runs
    - Entry point for defining robot behavior
    - Location: `src/ssl/grSim/sandbox_process.py`
    - This is where you will implement or modify strategies and behaviors for the robots.

### Important and useful functions/classes
**Coordinate transformation**

Import like this:
`from team_control.world.transform_cords import world2robot`


What it does:
```
def world2robot(robot_position,target_position):
    '''
        input:
            Target_position: position in the world coordinate system (x,y)
            robot_position: robot pose (x, y, theta)
        output:
            t: target position in respect to robot coordinate system (x,y)
    '''
```
**RobotCommand**

Import like this:
`from team_control.network.robot_command import RobotCommand`

What it does:
```
class RobotCommand():
    """
    RobotCommand represents the desired motion command for a single robot.

    Attributes:
        robot_id (int): Unique robot identifier.
        vx (float): Velocity along the X-axis (m/s).
        vy (float): Velocity along the Y-axis (m/s).
        w (float): Angular velocity (rad/s).
        kick (int): Kick flag (0 = no kick, 1 = kick).
        dribble (int): Dribble flag (0 = off, 1 = on).
        (time_origin (float): Original creation time of the packet.)
        (time_set (float): Timestamp when this RobotCommand object was instantiated.)
    """
```

**Spawning a ball**

Import like this:
`from team_control.ball.send_ball_grsim import send_ball_to_grsim`

What it does:
```
def send_ball_to_grsim(x, y, vx, vy, address="127.0.0.1", port=20011):
    """
    Spawn or move the ball in grSim to position (x, y)
    with velocity (vx, vy).

    Args:
        x (float): X position in mm.
        y (float): Y position in mm.
        vx (float): Velocity in X direction in m/s.
        vy (float): Velocity in Y direction in m/s.
    """
```

## Running Code in grSim
1. Run grSim `./bin/grSim` (in one terminal)
2. Check that the “vision port” in sandbox.py is set to the same value as “Vision multicast port” in grSim
3. Check that the “CMD_LISTEN_PORT” in sandbox_process.py is set to the same value as “Command listen port” in grSim
4. Run `sandbox.py` (in another terminal) -->