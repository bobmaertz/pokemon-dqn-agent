
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pyboy

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
            steps_per_episode=10000):
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

