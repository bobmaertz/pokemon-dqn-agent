# Pokémon Blue Reinforcement Learning

A deep reinforcement learning project that trains an AI agent to play Pokémon Blue using PyBoy emulation and deep Q-learning.

## Overview

This project uses deep Q-learning to train an agent to navigate and play Pokémon Blue on the Game Boy. The system uses PyBoy for emulation and PyTorch for the deep learning model.

Inspired by [Peter Whidden's Youtube Video](https://www.youtube.com/watch?v=DcYLT37ImBY&t=156s).

## Features

- Custom Gym environment for Pokémon Blue
- Deep Q-Learning implementation with PyTorch
- Exploration-based reward system
- Automated training pipeline
- Model checkpointing

## CI

GitHub Actions runs on every push and pull request:

- `ci / lint` (ruff)
- `ci / test` (pytest)

To make lint + tests required for each commit to your default branch, enable a branch protection rule in GitHub and require these status checks.

## Prerequisites

- Python 3.8+
- PyTorch
- Gymnasium
- PyBoy
- NumPy
- A legally obtained Pokémon Blue/Red ROM file

## Installation
1. Clone repo
2. Install dependencies:

```sh
python3 -m pip install -r requirements.txt
```

3. (Optional) Install dependencies for visualization/analysis (`analysis.py`):

```sh
python3 -m pip install -r requirements-analysis.txt
```

4. Place your legally obtained Pokémon ROM file in the project directory as `POKEMONR.GBC`.

5. (Optional) Create a saved state file using PyBoy and save it under `env_state/`.

## Usage
Run the runner:

```sh
python3 pokemon_blue_agent.py --rom_path ./POKEMONR.GBC --state_file ./env_state/game_start.state --steps 500
```

Write per-episode step logs (for parsing with `analysis.py`):

```sh
python3 pokemon_blue_agent.py --rom_path ./POKEMONR.GBC --state_file ./env_state/game_start.state --steps 5000 --episode_log_dir ./episode_logs
```

## How It Works

### Environment

The custom `PokemonBlueEnv` class wraps PyBoy to provide a Gym-compatible environment. It:
- Captures screen state as observations
- Translates actions to Game Boy button presses
- Calculates rewards based on exploration of new map tiles
- Manages episode termination

### Agent

The `DeepQLearningAgent` implements deep Q-learning with:
- Convolutional neural network for processing screen images
- Experience replay for stable learning
- Epsilon-greedy exploration strategy
- Target network for stable Q-value estimation

### Reward System

The agent receives rewards for:
- Exploring new map tiles (visiting new coordinates)

## Configuration

Modify these constants in the script to adjust training:

- `REPLAY_MEMORY_SIZE`: Size of experience replay buffer
- `STEPS_PER_EPISODE`: Maximum steps per episode
- `NUM_EPISODES`: Number of episodes to train

## Results

After training, model weights are saved to:
- `[MODEL_NAME]_model_state_dict.pth`
- `[MODEL_NAME]_optimizer_state_dict.pth`

## Future Improvements

- Incorporate game-specific rewards (badges, Pokémon caught, etc.)
- Implement prioritized experience replay
- Add visualization tools for agent behavior
- Support for saving/loading training progress

## Acknowledgments

- Built with the guidance of Claude AI
- Inspired by Peter Whidden's reinforcement learning projects
- Uses PyBoy for Game Boy emulation