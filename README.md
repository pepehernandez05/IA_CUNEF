# Basketball Free-Throw Agent — Tabular Q-Learning

A Reinforcement Learning project developed at **CUNEF Universidad** as part of the Computer Science Engineering degree by José Ignacio Hernández Pablos and Rodrigo López Rodríguez.

The project implements a tabular Q-Learning agent that learns to score free throws in a custom 2D basketball simulation. The agent selects a launch angle and initial velocity to adapt to different basket positions.

## Features

- Custom basketball environment based on Gymnasium.
- Tabular Q-Learning with an epsilon-greedy exploration policy.
- Discretization of continuous actions into 25 possible combinations.
- Reward shaping to modify the learning signal.
- Model evaluation over 1,000 episodes.
- Pygame visualization of the trained agent.
- Training logs and learning-curve generation.

## Project Structure

```text
IA_CUNEF/
├── README.md
├── requirements.txt
├── .gitignore
├── agente.py
├── basket_env.py
├── ejecutar_agente.py
├── evaluar_agente.py
├── best_model/
│   └── basket_agent_q_table.npy
├── logs/
│   └── curva_aprendizaje.png
└── wrappers/
    ├── __init__.py
    ├── env_wrappers.py
    ├── log_utils.py
    └── render_utils.py
```

## Getting Started

### Requirements

- Python 3.8 or later
- The dependencies listed in `requirements.txt`

Install the dependencies:

```bash
pip install -r requirements.txt
```

### 1. Train the Agent

```bash
python agente.py
```

The training loop runs for 500,000 steps. The resulting Q-table is saved to:

```text
best_model/basket_agent_q_table.npy
```

Training logs and the learning curve are stored in the `logs/` directory.

### 2. Evaluate the Agent

```bash
python evaluar_agente.py
```

The evaluation script runs 1,000 episodes with exploration disabled and reports the agent's performance.

### 3. Visualize the Trained Agent

```bash
python ejecutar_agente.py
```

Launches the agent in a Pygame visualization, allowing its shots to be observed in the basketball simulation.

### 4. Regenerate the Learning Curve

```bash
python -c "from wrappers.log_utils import plot_logs; plot_logs('./logs/')"
```

This command uses the training logs to generate a learning-curve plot. Run training first if the required log file is missing.

## Technical Overview

### Environment

The custom environment simulates projectile motion in two dimensions. The ball is launched with an angle and initial velocity, and the simulation checks whether its trajectory intersects the basket.

- **Observation space:** discrete states representing basket positions.
- **Action space:** continuous angle and velocity values, discretized by the agent.
- **Physics:** gravitational acceleration of 9.8 m/s² and simulation steps of 0.1 seconds.
- **Episode limit:** 30 simulation steps per launch.

### Action Discretization

The agent considers five angle levels and five velocity levels, producing 25 discrete actions. Each action index maps to one angle-velocity combination within the environment's action-space limits.

The Q-table stores an estimated value for each state-action pair.

### Learning Algorithm

The agent uses tabular Q-Learning with an epsilon-greedy policy. During training, it balances exploration of new actions with exploitation of actions that have previously received higher Q-values.

The main training parameters are:

| Parameter | Value |
|---|---:|
| Learning rate | 0.1 |
| Discount factor | 0.99 |
| Initial epsilon | 1.0 |
| Minimum epsilon | 0.01 |
| Epsilon decay | 0.9999 |
| Training steps | 500,000 |

### Reward Shaping

The `RewardShapingWrapper` multiplies the environment's reward by 1.2. This scales both positive rewards and penalties and changes the magnitude of the learning signal.

## Possible Improvements

- Explore Deep Q-Networks or other methods for larger state spaces.
- Investigate finer action discretization.
- Allow actions to change during the ball's trajectory.
- Evaluate performance across multiple training seeds and compare learning stability.
