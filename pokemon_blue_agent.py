'''
Agent and Environment for Pokemon Blue env
'''

import random
from collections import deque, namedtuple
import os
import argparse

import wandb
import gymnasium as gym
import numpy as np
import pyboy
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces


REPLAY_MEMORY_SIZE = 400
STEPS_PER_EPISODE = 10000

# Path to legally obtained Pokémon ROM
ROM_PATH = './POKEMONR.GBC'
STATE_FILE = './env_state/game_start.state'
MODEL_NAME = "pokemon_blue_dqn"

ACTION_MAP = {
    0: 'up',
    1: 'down',
    2: 'left',
    3: 'right',
    4: 'A',
    5: 'B',
    # 6: 'start',
    # 7: 'select'
}

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class PokemonBlueEnv(gym.Env):
    """
    Custom Gymnasium environment for Pokémon Blue
    to facilitate deep learning training
    """

    def __init__(
            self,
            rom_path,
            state_file=None,
            render_mode="null",
            emulation_speed=0,
            steps_per_episode=STEPS_PER_EPISODE):
        super().__init__()

        self.rom_path = rom_path
        self.emulation_speed = emulation_speed
        self.render_mode = render_mode
        self.state_file = state_file
        self.steps_per_episode = steps_per_episode
        self._current_state = None
        self.screen_memory = []
        self.steps = 0
        self.explore_map = {}

        # Initialize PyBoy emulator
        self.pyboy = pyboy.PyBoy(
            self.rom_path,
            window=self.render_mode,
            sound=False)
        self.pyboy.set_emulation_speed(self.emulation_speed)

        # Load saved state if set
        self.load_saved_state()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(ACTION_MAP))

        # Observation space: screen pixels and game state
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(1, 144, 160),  # Game Boy screen dimensions
            dtype=np.uint8
        )

    def load_saved_state(self):
        """
        Load the saved state for the environment
        """
        if self.state_file:
            with open(self.state_file, "rb") as f:
                self.pyboy.load_state(f)

    def step(self, action):
        """
        Execute one time step within the environment

        Args:
            action (int): Action to take in the environment

        Returns:
            observation (ndarray): Agent's observation of the environment
            reward (float): Amount of reward returned after previous action
            terminated (bool): Whether the episode has ended
            truncated (bool): Whether the episode was truncated
            info (dict): Additional diagnostic information
        """
        # Translate action to PyBoy input
        self._take_action(action)

        # Advance emulator frame
        self.pyboy.tick(24, render=False, sound=False)
        self.pyboy.tick(1, render=True, sound=False)

        # Capture screen state
        screen = self._get_screen()

        # Compute reward (to be refined based on game mechanics)
        reward = self._compute_reward()

        # Remember scene for later rewards.
        # self.screen_memory.append(screen)

        # Check for episode termination
        terminated = self._is_episode_done()

        # Additional info for debugging/analysis
        info = self._get_game_state()

        return screen, reward, terminated, False, info

    def _take_action(self, action):
        """
        Translate action to PyBoy input
        """
        self.pyboy.button(ACTION_MAP[action])

    def _get_screen(self):
        """
        Capture and process game screen

        Returns:
            numpy array of screen pixels
        """
        screen = self.pyboy.screen.image
        gray_screen = screen.convert('L')  # Convert to grayscale
        screen_array = np.array(gray_screen)
        screen_array = screen_array.reshape((1, 144, 160))
        return screen_array

    def _compute_reward(self):
        """
        Compute reward based on game state

        This is a placeholder and should be customized based on
        specific training objectives
        """
        map_num = self.pyboy.memory[0xD35E]
        x_coord = self.pyboy.memory[0xD361]
        y_coord = self.pyboy.memory[0xD362]
        loc = f"{map_num}:{x_coord}:{y_coord}"

        if loc in self.explore_map:
            return 0
        else:
            self.explore_map[loc] = True
            return 1.0
        return 0

    def _is_episode_done(self):
        """
        Determine if the current episode is finished

        Returns:
            bool: Whether episode is terminated
        """

        self.steps = self.steps + 1
        if self.steps > self.steps_per_episode:
            return True
        # Check for game over conditions
        return False

    def _get_game_state(self):
        """
        Extract relevant game state information

        Returns:
            dict: Game state metrics
        """

        # Extract memory values, player position, etc.
        return {}

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to initial state

        Returns:
            Initial observation, info dict
        """
        super().reset(seed=seed)

        self.screen_memory = []
        self.steps = 0
        self.explore_map = {}

        # Perform a complete stop
        self.pyboy.stop(save=False)

        # Reinitialize PyBoy from scratch
        self.pyboy = pyboy.PyBoy(
            self.rom_path,
            window=self.render_mode,
            sound=False)
        self.pyboy.set_emulation_speed(self.emulation_speed)

        # Reload from our saved state
        self.load_saved_state()

        initial_screen = self._get_screen()
        return initial_screen, {}

    def render(self):
        """
        Render the environment
        """
        if self.render_mode == 'human':
            self.pyboy.screen.image.show()

    def close(self):
        """
        Close the environment
        """
        self.pyboy.stop()


class DeepQLearningAgent:
    """
    Deep Q-Learning agent for Pokémon Blue
    """

    def __init__(
            self,
            state_size,
            action_size,
            replay_memory_size=REPLAY_MEMORY_SIZE):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_memory_size = replay_memory_size
        self.replay_memory = deque(maxlen=self.replay_memory_size)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001
        self._train_counter = 0

        if torch.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Neural Network for Q-learning
        self.policy_model = self._build_model()
        self.policy_model.to(self.device)

        self.target_model = self._build_model()
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.target_model.to(self.device)

        self.optimizer = optim.AdamW(
            self.policy_model.parameters(),
            lr=self.learning_rate,
            amsgrad=True)

    def _build_model(self):
        """
        Create Deep Neural Network for Q-Learning
        """
        model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(64 * 36 * 40, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_size)
        )

        return model

    def act(self, state):
        """
        Choose action using epsilon-greedy strategy
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(
            state).unsqueeze(0).to(device=self.device)
        q_values = self.policy_model(state_tensor)
        return torch.argmax(q_values).item()

    def update_memory(self, transition):
        """
        Update the replay memory with the latest transition tuple
        - state, action, reward, next_state, done
        """
        self.replay_memory.append(transition)

    def update_target_network(self):
        """
        Update the target network with the policy network weights
        """
        self.target_model.load_state_dict(self.policy_model.state_dict())

    def train(self):
        """
        Train the agent using Deep Q-Learning
        """
        # Dont want to train on memory less than REPLAY_MEMORY_SIZE, not a big
        # enough batch.
        if len(self.replay_memory) < self.replay_memory_size:
            return

        minibatch = random.sample(self.replay_memory, 64)

        # Reviewing algorithm from https://www.youtube.com/watch?v=qfovbG84EBg&t=335s
        # TODO: Double check normalization of 255
        current_states = torch.FloatTensor(
            np.array([transition.state for transition in minibatch]) / 255).to(self.device)
        actions = torch.LongTensor(
            [transition.action for transition in minibatch]).to(self.device)
        rewards = torch.FloatTensor(
            [transition.reward for transition in minibatch]).to(self.device)
        next_states = torch.FloatTensor(np.array(
            [transition.next_state for transition in minibatch]) / 255).to(self.device)
        dones = torch.FloatTensor(
            [transition.done for transition in minibatch]).to(self.device)

        # Compute Q-values for current states
        curr_q = self.policy_model(current_states)
        curr_q = curr_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute Q-values for next states using target network
        next_q = self.target_model(next_states).detach()
        max_next_q = next_q.max(1)[0]

        # Compute target Q-values
        target_q = rewards + (self.gamma * max_next_q * (1 - dones))

        # Compute loss
        loss = nn.MSELoss()(curr_q, target_q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        if self._train_counter % 250 == 0:
            self.update_target_network()

        self._train_counter += 1

    def save(self, name):
        """
        Save model weights
        """
        # Save the model's state_dict
        model_filename = f"{name}_model_state_dict.pth"
        torch.save(self.target_model.state_dict(), model_filename)

        optimizer_filename = f"{name}_optimizer_state_dict.pth"
        torch.save(self.optimizer.state_dict(), optimizer_filename)


def main():
    '''
    Main function to train a Deep Q-Learning 
    agent on the Pokémon Blue environment.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--replay_memory_size',
        type=int,
        default=REPLAY_MEMORY_SIZE,
        help='Replay memory size')
    parser.add_argument(
        '--steps_per_episode',
        type=int,
        default=STEPS_PER_EPISODE,
        help='Steps per episode')
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=10,
        help='Number of episodes')
    parser.add_argument(
        '--rom_path',
        type=str,
        default=ROM_PATH,
        help='Path to Pokémon ROM')
    parser.add_argument(
        '--state_file',
        type=str,
        default=STATE_FILE,
        help='Path to emulator state file')
    parser.add_argument(
        '--model_name',
        type=str,
        default=MODEL_NAME,
        help='Model name for saving')
    parser.add_argument(
        '--wandb_entity',
        type=str,
        default=os.environ.get(
            'WANDB_ENTITY',
            ''),
        help='Weights & Biases entity')
    parser.add_argument(
        '--wandb_project',
        type=str,
        default=os.environ.get(
            'WANDB_PROJECT',
            ''),
        help='Weights & Biases project')
    args = parser.parse_args()

    num_episodes = args.num_episodes
    model_name = args.model_name
    # Create environment
    env = PokemonBlueEnv(args.rom_path, args.state_file)

    # Initialize agent
    agent = DeepQLearningAgent(
        state_size=env.observation_space.shape,
        action_size=env.action_space.n,
        replay_memory_size=args.replay_memory_size
    )

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        config={
            "learning_rate": agent.learning_rate,
            "device": str(agent.device),
            "epsilon": agent.epsilon,
            "epsilon_decay": agent.epsilon_decay,
            "gamma": agent.gamma,
            "batch_num": 64,
            "replay_memory_size": agent.replay_memory_size,
            "architecture": "CNN",
            "epochs": num_episodes,
        },
    )
    # Training loop
    for episode in range(num_episodes):
        state, _ = env.reset(options={"initial_run": True})
        done = False
        cumulative_reward = 0

        train_count = 0
        while not done:
            train_count += 1

            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)

            agent.update_memory(
                Transition(
                    state,
                    action,
                    reward,
                    next_state,
                    done))

            if train_count % 4 == 0:
                agent.train()
                train_count = 0

            state = next_state
            cumulative_reward += reward

        run.log({'reward': cumulative_reward, 'episode': episode})
        # Decay exploration rate
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

    agent.save(model_name)

    run.finish()


if __name__ == '__main__':
    main()
