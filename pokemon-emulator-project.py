import random
from collections import deque

import gymnasium as gym
import numpy as np
import pyboy
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from pyboy.utils import WindowEvent

REPLAY_MEMORY_SIZE = 500
STEPS_PER_EPISODE = 10000
NUM_EPISODES = 100
MODEL_NAME = "255_8_Initial"

# Path to legally obtained Pokémon ROM
ROM_PATH = './POKEMONR.GBC'
STATE_FILE = './state_file.state'
class PokemonBlueEnv(gym.Env):
    """
    Custom Gymnasium environment for Pokémon Blue 
    to facilitate deep learning training
    """
    def __init__(self, rom_path, state_file=None, render_mode=None):
        super().__init__()

        # Initialize PyBoy emulator
        self.pyboy = pyboy.PyBoy(rom_path,window="null")
        self.pyboy.set_emulation_speed(16)

        self.state_file = state_file
        self.load_saved_state()
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
        self.steps = 0 
        self.explore_map = {}

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
        # self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_SELECT)
        # self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)

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
        # elif actions[action] == 'start':
        #     self.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
        # elif actions[action] == 'select':
        #     self.pyboy.send_input(WindowEvent.PRESS_BUTTON_SELECT)
    
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

        map_num = self.pyboy.memory[0xD35E]
        x_coord = self.pyboy.memory[0xD361]
        y_coord = self.pyboy.memory[0xD362]
        loc = f"{map_num}:{x_coord}:{y_coord}"
        
        try:
            self.explore_map[loc]
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
        if self.steps > STEPS_PER_EPISODE:
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
        self.close()

        if options is None or not options["initial_run"]:
            # Reset PyBoy to start of game
            self.pyboy.game_wrapper.reset_game()

        ## Reload from our saved state
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
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

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

        self.optimizer = optim.AdamW(self.policy_model.parameters(), lr=self.learning_rate, amsgrad=True)

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
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device=self.device)
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

        # Dont want to train on memory less than REPLAY_MEMORY_SIZE, not a big enough batch. 
        if len(self.replay_memory) < REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, REPLAY_MEMORY_SIZE)
        
        ## Reviewing algorithm from https://www.youtube.com/watch?v=qfovbG84EBg&t=335s
        #TODO: Double check normalization of 255 
        current_states = torch.FloatTensor(np.array([transition[0] for transition in minibatch])/255).to(self.device)
        actions = torch.LongTensor([transition[1] for transition in minibatch]).to(self.device)
        rewards = torch.FloatTensor([transition[2] for transition in minibatch]).to(self.device)
        next_states = torch.FloatTensor(np.array([transition[3] for transition in minibatch])/255).to(self.device)
        dones = torch.FloatTensor([transition[4] for transition in minibatch]).to(self.device)
       
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
        self.update_target_network()

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
    # Create environment
    env = PokemonBlueEnv(ROM_PATH, STATE_FILE)

    # Initialize agent
    agent = DeepQLearningAgent(
        state_size=env.observation_space.shape,
        action_size=env.action_space.n
    )
   
    # Training loop
    for episode in range(NUM_EPISODES):
        state, _ = env.reset(options={"initial_run":True})
        done = False
        total_reward = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            
            agent.update_memory((state, action, reward, next_state, done))
            
            # Train agent
            agent.train()
            
            state = next_state
            total_reward += reward

        print(f"Episode {episode}: Total Reward = {total_reward}")
        
        # Decay exploration rate
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

    agent.save(MODEL_NAME)

if __name__ == '__main__':
    main()
