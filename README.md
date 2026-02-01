# RoboCup Final Competition - BitKickers
The following document describes our teams implementation for the RoboCup SSL lab final competition. Throughout the semester, we have worked on separate assignments such as goalie, striker, obstacle avoidance, passing and interceptor. In this final competition we have brought all of them together and made a single match-ready code base that is able to perform all of the above mentioned skills together.

This final competition will be a simplified version of a RoboCup tournament, without the full complexity of official SSL rules (e.g., ball placement and other referee-controlled procedures).

Each team plays with 3 robots in a match (3 vs 3). Plus an additional goalie.

This document includes the information about the competition, rules and restrictions under which we are playing. It'll also define all our design choices, technical implementation and models.

## Match Rules

- Robots per side: 3 robots (simulated) plus one goalie.
- Match duration: 6 minutes of running time per match (continuous).
- Match start: The match starts when the ball is spawned in the center of the field at position (0, 0).
- Field: Standard Division B field dimensions as used throughout the course. The ball must remain in-field at all times.
- Autonomy: Game play must be fully autonomous. This means that you must not touch the laptop executing the code while the game is running.
- Ball Out of Play and Goals: Due to the simplified game rules, no distinction is made regarding which team caused the ball to leave. If the ball leaves the field of play, or if a goal is scored, the ball will be respawned in a random position within the center area of the field.

### Fouls
- Defense Area Violations: Robots may not enter either their own or the opponent’s defense area.
- Collisions and Unsafe Behavior: Robots may make light contact, but high-speed or forceful collisions are not allowed. It is the referee’s responsibility to determine whether a collision constitutes a crash and to decide which team is at fault.

**Note**: For every three fouls committed by a team, one goal will be awarded to the opposing team.


### Assumptions:
Our assumptions is that we're always playing on the right side of the field. If we are supposed to play on the left side, then we mathematically change the model inputs. Our variable is_team_yellow throughout this project can be equally interpreted as if the team is playing on the right side.
Our model has the assumption that the robot belongs to the right side of the field and plays on the right-side. Hence, if the input data belongs to left side players we transform the info to the projection of the right side, and then transform the commands back to the left side.

------------------------------------------------------------------------------

# BitKickers' Gameplay

## Manual Controller
In order to be able to use what we have developed throughout the semester, we will first use our manual controller.

The strategy is the following:
We will have 3 field players and one goalie. We divide the field into four sections shown in the image below:
![Division of the field by the role](docs/field_role_division.png?raw=true "Field-Role division")

The goalie will be inside the defense area and on alert for the incoming balls. The defender will be in the section in front of the defense area and behind the penalty point inside our own field. The midfielder will be placed inside the section between two penalty points. And the striker is based from the opponents penalty point and behind the opponent's defense area.

The strategy the players have is the following:
For each of the players, there is two different big cases. If we have the ball or not.
1. Agent has the ball inside its area
    - if the agent has the ball in its possession it should either pass it or kick based on the situation.
        - If it is a midfielder or a striker and there exists a clear shot to a point in the goal, we shoot the ball with maximum power to that point.
        - If we don't have a clear shot, we will consider 100 points in the field, towards the opponents goal distributed with equal distance in the field. The points in behind the agents we discard, because we simply don't want to play backwards. The chosen points will be used for passing. Then we use what we call a point_scoring function, this function assigns scores to these points based on the situation of the points. Criterias are the following:
        1- the distance of the point to me (the closer the point we want to pass the ball is better)
        2- the distance of the closest teammate to the point, the closer the better.
        3- the distance of the closest opponent to the point, the further the opponent the better.
    After scoring all the points we pass the ball to the point with the highest score.
    ![point scoring](docs/point_scoring.png?raw=true "Point scoring")
    The point_scoring function will receive the information of the ball, the teammates, the opponents and the robot issuning the command and returns the point the ball should be sent to. There is a rough sketch of it above.
2. Opponent has the ball inside agents area:
    the agent will be going toward the opponent with caution so that it will not collide and cause a foul. In case we can get the ball, we turn on the agents dribbler so it gets the ball from the opponent. This is included in our interceptor process.
3. Ball is inside the agents area, but no one holds it:
    in this case the agent will move towards the ball with its kicker faced toward the ball so that it can possess it and activate case 1.

How they'll act is the following:
- The goalie will use it's trajectory prediction to predict the path of the incoming balls. It will align itself and tries to catch the ball. When the goalie has the ball, it will see it's trajectory to the defender. If it's clear, it'll pass the ball to the defender, if the trajectory is not clear, it’ll see its trajectory to the midfielder and pass the ball to it. If not, it’ll see the trajectory to the striker. If clear pass the ball to the striker. If the goalie has no clear pass oppurtunity to any of the teammates it will hold the ball shortly and lets the teammates move themselves so that a trajectory for a path clears up.
- ⁠The defender will act as the interceptor when it doesn't have the ball and the opponent is inside that part of the field with the ball. It will pressure them and try to block their shot to the goal. If it has the ball it can check for clear trajectories to the agents in front of it(like the goalie situation) and pass the ball to them. If not clear, it can also shoot the ball with maximum power towards the opponents goal, in a place that their goalie is not there to maximize our chances of scoring.
- ⁠The midfielder will have only the striker to pass to, and the opponent goal to shoot to when having the ball. When it doens't own the ball, and the opponents are inside its area and having the ball, ot will also be an interceptor and try to pressure them.
- ⁠The striker will just do obstacle avoidance while having the dribble flag ON, not to lose the ball, it'll move inside it's area and when having a clear trajectory to a point in opponents goal, aligns itself and takes the shot with maximum power(which is 10) like we already have in the striker_process.py, like the other agents, it'll try to put pressure and block the passing of the opponents given the ball is inside its part of the field.

We note here that the maximum kick power of the agents is 10. For the kick power, based on what we want to do, we change the power. If we are passing we use 10 * normalized distance in our calculations. When kicking the ball towards the opponents goal, we set kick power = 10 in order to maximize chances of scoring.

Then after developing this method, we have used this process and checked that our approach works good. The next section describes our RL implementation.

------------------------------------------------------------------------------

# RL
In order for our agents to benefit the power of reinforcement learning, we will first integrate our manual controller in our model, using imitation learning.


Using the ideas of already hugely successful approaches like AlphaGo, AlphaZero and OpenAI Five, we decided to use the PPO algorithm for a multiagent setting. Since we are in a strategical environment, off-policy learning doesn't make sense. Because off-policy is used in environments that the model is trying to learn physics not strategy.

Here is a short comparison of this design choice:

*Off-policy vs on-policy?*

We cannot use off-policy learning in this setting since we are doing self-play. Suppose we're using off-policy for example and we have a replay buffer. Replay buffer has actions from previous policies from which we have diverged, so we're basically updating on trash data. That's why we have to be on-policy. The difficulty with on-policy methods is that they're starting off cold. This problem we take care of using imitation learning and training our model on the manual controller data.

Think of off-policy as: “Learning a value/policy function for a fixed world.”

But think of on-policy as: “Learn a strategy in a changing world.”

Usually in reinforcement learning settings off-policy methods are used for physics problems in which *The world does not care that the agent is learning*.

For strategy problems like RoboCup SSL context, *The world changes because the agent is learning*.

This explanation removes the off-policy options such as TD3, DDPG and etc. We base our approach on the multiagent version of the Proximal Policy Optimization algorithm.

## Imitation learning
We use the strategy and the data of our implementation to have a baseline for our agents.
Imitation learning will enable our model to imitate the strategy and the commands that we have developed for our agents. Then after the model is trained on a good amount of running time and the supervised learning tells us the loss has dropped, it means now our model is able to imitate our strategy and we can move on to the self-play and multiagent reinforcement learning.

Without supervised learning in the beginning based on what we have done in the goalie, striker, passing interceptor and mover phases we have to let the robots self-play to learn the physics of the model and the basic skills, so our curriculum learning would take a long time, but with supervised learning with datasets created from our data it already knows some of the skills, like moving, rotating, kicking and dribbling and some team cooperative play, which puts us ahead hugely. Then only using self-play and reinforcement learning the model learns adverserial play with opponents, better cooperative team-play and more strategic behavior. This also helps us to reduce our training time. If we haven't had used this approach, we had to first let a single robot roam around in a small field, with very sparse rewards, to get to the ball, to kick the ball, to learn to rotate and dribble and so on. And then when it learned this playing style we needed to add agents and adverseries in each step. Also we had to increase our episode length. So our curriculum would have multiple axes. For example time(episode length), number of teammates, opponents, field size and so on would be our curriculum axes. Also we note here that since we don't have an environment that we can get the next states deterministicly(for example like OpenAI gym environments like mountain car, cart pole and etc.) we have to get the frames and the data from grSim. This physical simulator introduces a heavy bottleneck which increases the training time orders of magnitude. Using imitation learning we overcome this bottleneck to some extent.

In the image below you can see the result of the supervised learning of our actor model on the manual controller implementation:
![Supervised Imitation Learning](docs/supervised_imitation_loss.png?raw=true "Imitation loss")


## Deep Reinforcement learning

Here we will describe our neural network, the design choices and so on.

### Attention mechanism

### LSTM
We need to have a memory of the agents and their current velocity and status as well. Because grim does not give us the current directions and angular velocity of the robots, it is necessary that we have a Memory mechanism inside our model.

### Encoder
Both of the models will be using the same encoder, hence we train the encoder on a lot of data, freeze and then store it and give it to the models.

### Plasiticity loss and the twohot encoding idea
### Two hot encoding


## Rewards
The core part of having a sound RL implementation is to give agents cleverly engineered rewards. We also need to take care of the sparse reward setting. In RL settings like RoboCup that we have sparse rewards like scoring a goal + or conceding a goal - it will hugely benefit us if we have dense rewards for the robots.
So here are the rewards we will give the players and how they affect our training.
- A
- B
- C

Also there are approaches that can take care of the rewards but we will not include them here since they would increase the complexity of the approach and due to deadline and time restrictions we wanted to have a working system without increasing complexity orders of magnitude.
- We introduce dense rewards in our environment (distance to the ball, goals, facing the ball, collisions, ...)
- Methods like HER(hindsight experience replay), curiousity driven would increase complexity. Our goal is not always straightforward so we cannot use HER method directly, curiousity driven RL also need at least 3 more networks and since our settings is pretty straightforward the dense reward setting that we created is enough. Also HER is for the off-policy settings and we're using on-policy.

Another idea we have used is the team spirit. If we don't use team spirit and divide the rewards between personal rewards and team rewards it's possible that one robot is doing extremely bad but because one striker is doing really good and scoring everytime, the idle robot gets rewarded as well. So we have to find a sweet spot between the personal and team rewards like the OpenAI five team did in their Dota2 multiagent implementation.


------------------------------------------------------------------------------

# GrSim:
Running and getting grSim to work will be a straightforward process on linux and older Mac systems with intel processors. But it will be not so easy on ARM series. Both because of the old resources that the grSim uses(such as old cmake, ode, etc), and also because of the macos underlying network structure and how it handles the multicasting. We have tried to explain our approach here in the following so people can use this data to use it to run it on mac in the most straightforward way possible.

To install grSim on Mac with M processors follow these steps:

**Important note**: grSim will be compiled with CMake < 4, if you have CMake >= 4 you have to download v3.X.X matching your arch from this link[https://github.com/Kitware/CMake/releases].
Here we use cmake-3.28.3[https://github.com/Kitware/CMake/releases/tag/v3.28.3]

brew install pkg-config
brew install ode
brew install qt@5
**Note**: as grSim uses a very old version of protobuf/vartypes and brew doesn't have it, skip the installation so the CMake downloads and builds the required version.

```bash
git clone git@github.com:RoboCup-SSL/grSim.git
cd grSim/
mkdir build
cd build
```

./path/to/your/cmake -DCMAKE_INSTALL_PREFIX=/opt/homebrew/Cellar ..
**Note**: in the grSim offical install instructions that is denoted as /usr/local which is the path on intel based machines, for installation with homebrew use /opt/homebrew/Cellar

After you install it cd ../bin/ and ./grSim to run it. (or if you want just cp it to /opt/homebrew/bin so its visible from the PATH)

**Issue**: you may see the following message repeatedly in the console log "Sending UDP datagram failed (maybe too large?). Size was: X byte(s)."
this issue is related to how MacOS handles packet sending on multicast mode, to resolve this issue set the 'Vision multicast address' to the address of your system in the interface you are using. For example if you're using wifi and network interface is "en0" just ifconfig en0 and then copy your systems address in that network interface.

### Ubuntu/VM
sudo apt install git build-essential cmake pkg-config qtbase5-dev libqt5opengl5-dev libgl1-mesa-dev libglu1-mesa-dev libprotobuf-dev protobuf-compiler libode-dev libboost-dev
sudo apt install tini libqt5opengl5 libode8 x11vnc xvfb
cd grSim/
mkdir build
cd build/
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
make
cd ../..

### grSim headless
In order to run grSim on VMs, since they don't have display we have to choose the headless version:
```bash
grSim -H -platform offscreen
```

------------------------------------------------------------------------------
# Running the code

In order to run our code, first create a virtual environment using python3.11 and higher.
```bash
python3 -m venv .venv
```

Then we will have to install the pip packages and our development project.
To do so, you will first have to activate the environment.
In Linux :
```bash
source .venv/bin/activate
```

### Install necessary modules
Then install the modules and this project as *Editable* project (that uses `pyproject.toml`)
```bash
pip install -e .
```
------------------------------------------------------------------------------
