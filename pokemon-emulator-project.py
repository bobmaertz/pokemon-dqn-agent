import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pyboy
import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque 

import random
from pyboy.utils import (
    WindowEvent,
)
class PokemonBlueEnv(gym.Env):
    """
    Custom Gymnasium environment for Pokémon Blue 
    to facilitate deep learning training
    """
    def __init__(self, rom_path, state_file=None, render_mode=None):
        super().__init__()

        # Initialize PyBoy emulator
        self.pyboy = pyboy.PyBoy(rom_path)
       
        if state_file: 
            with open(state_file, "rb") as f:
                self.pyboy.load_state(f)
                
        # Define action and observation spaces
        # Actions could include:
        # 0: Up, 1: Down, 2: Left, 3: Right, 4: A Button, 5: B Button, 6: Start, 7: Select
        self.action_space = spaces.Discrete(8)
        
        # Observation space: screen pixels and game state
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(1, 144, 160),  # Game Boy screen dimensions
            dtype=np.uint8
        )
        
        self.render_mode = render_mode
        self._current_state = None
        self.screen_memory = []
        
        self.explore_map = {}

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
        self.pyboy.tick()
        
        # Capture screen state
        screen = self._get_screen()

        # Compute reward (to be refined based on game mechanics)
        reward = self._compute_reward()
        
        # Remember scene for later rewards. 
        self.screen_memory.append(screen)

        # Check for episode termination
        terminated = self._is_episode_done()
        
        # Additional info for debugging/analysis
        info = self._get_game_state()
        
        return screen, reward, terminated, False, info
    
    def _take_action(self, action):
        """
        Translate action to PyBoy input
        """
        actions = {
            0: 'up',
            1: 'down', 
            2: 'left',
            3: 'right',
            4: 'A',
            5: 'B',
            6: 'start',
            7: 'select'
        }
        
        # Reset previous inputs
        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_UP)
        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN)
        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT)
        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_B)
        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_SELECT)
        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)

        # Apply selected action
        if actions[action] == 'up':
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_UP)
        elif actions[action] == 'down':
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
        elif actions[action] == 'left':
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
        elif actions[action] == 'right':
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
        elif actions[action] == 'A':
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
        elif actions[action] == 'B':
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_B)
   
    def _get_screen(self):
        """
        Capture and process game screen
        
        Returns:
            numpy array of screen pixels
        """
        screen = self.pyboy.screen.image
        gray_screen = screen.convert('L')  # Convert to grayscale
        screen_array = np.array(gray_screen).reshape(1, 144, 160)
        return screen_array
    
    def _compute_reward(self):
        """
        Compute reward based on game state
        
        This is a placeholder and should be customized based on 
        specific training objectives
        """
    

        # Inspiration for this came from: https://github.com/PWhiddy/PokemonRedExperiments/blob/master/v2/red_gym_env_v2.py#L334-L335
        # More Background: 
        #     - https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
        #     - https://datacrystal.tcrf.net/wiki/Pok%C3%A9mon_Red_and_Blue/RAM_map#Map_Header
        # reward = 0.0 

        mapNo = self.pyboy.memory[0xD35E]
        x_coord = self.pyboy.memory[0xD361]
        y_coord = self.pyboy.memory[0xD362]
        loc = f"{mapNo}:{x_coord}:{y_coord}"
        
        try:
            mem = self.explore_map[loc]
            return 0 
        except KeyError: 
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
        if self.steps > 10000:
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
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state
        
        Returns:
            Initial observation, info dict
        """
        super().reset(seed=seed)
        
        self.screen_memory = []
        self.steps = 0 
        # Reset PyBoy to start of game
        #self.pyboy.game_wrapper.reset_game()
        # self.pyboy.stop(False)
        # self.pyboy.game_wrapper.
        # # There should be an option here to reset to a saved state or to reset the game. 
        # if self.state_file: 
        #    with open(self.state_file, "rb") as f:
        #         self.pyboy.load_state(f)



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
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Neural Network for Q-learning
        self.model = self._build_model()
    
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

        #TODO: Add optimizer
        return model
    
    def act(self, state):
        """
        Choose action using epsilon-greedy strategy
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()
    
    def update_memory(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, done, next_state])
    
    def train(self, state, action, reward, next_state, done):
        """
        Train the agent using Deep Q-Learning
        """
        return 
        self.memory.append((state, action, reward, done, next_state))
        epsilon = math.pow(epsilon, 1/total_steps)
        minibatch = self.memory.sample(self.n_minibatch)
        for m_state, m_action, m_reward, m_done, m_next_state in minibatch:
            if m_done: 
                y = m_reward
            else:
                y =  m_reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            
            y_1 = self.model.predict(m_state)
            y_1[0][m_action] = y
            out = self.model.fit(m_state, y_1, verbose=0)

            total_loss += out.history.get('loss')[0]
         

def main():
    # Path to Pokémon Blue ROM (you'll need to provide this)
    ROM_PATH = './POKEMONR.GBC'
    STATE_FILE = './state_file.state'
    # Create environment
    env = PokemonBlueEnv(ROM_PATH, STATE_FILE)
    
    # Initialize agent
    agent = DeepQLearningAgent(
        state_size=env.observation_space.shape, 
        action_size=env.action_space.n
    )
    
    # Training loop
    num_episodes = 1
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            
            # Train agent
            agent.train(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
        
        print(f"Episode {episode}: Total Reward = {total_reward}")
        
        # Decay exploration rate
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

if __name__ == '__main__':
    main()
