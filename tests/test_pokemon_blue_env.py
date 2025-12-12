import importlib
import sys
import types

import numpy as np
import pytest
from PIL import Image


class FakeScreen:
    def __init__(self, image: Image.Image):
        self.image = image


class FakePyBoy:
    instances = []

    def __init__(self, rom_path, window=None, sound=False):
        self.rom_path = rom_path
        self.window = window
        self.sound = sound
        self._emulation_speed = None
        self.last_button = None
        self.tick_calls = []
        self.stop_calls = []
        self.load_state_calls = 0

        # Minimal memory interface used by PokemonBlueEnv
        self.memory = {}
        self.memory[0xD35E] = 0
        self.memory[0xD361] = 0
        self.memory[0xD362] = 0

        # Provide a valid 160x144 image
        rgb = np.zeros((144, 160, 3), dtype=np.uint8)
        self.screen = FakeScreen(Image.fromarray(rgb, mode="RGB"))

        FakePyBoy.instances.append(self)

    def set_emulation_speed(self, speed):
        self._emulation_speed = speed

    def load_state(self, file_obj):
        # Just record that load_state was requested.
        _ = file_obj.read(1)
        self.load_state_calls += 1

    def button(self, button_name):
        self.last_button = button_name

    def tick(self, frames, render=False, sound=False):
        self.tick_calls.append((frames, render, sound))

    def stop(self, save=False):
        self.stop_calls.append(save)


@pytest.fixture
def pokemon_blue_module(monkeypatch):
    """Import src.env.pokemon_blue with a fake pyboy module injected."""
    fake_pyboy = types.ModuleType("pyboy")
    fake_pyboy.PyBoy = FakePyBoy
    monkeypatch.setitem(sys.modules, "pyboy", fake_pyboy)

    module = importlib.import_module("src.env.pokemon_blue")
    module = importlib.reload(module)
    return module


def test_spaces_are_wired(pokemon_blue_module):
    env = pokemon_blue_module.PokemonBlueEnv(
        rom_path="/does/not/matter.gbc",
        state_file=None,
        render_mode="null",
        emulation_speed=0,
        steps_per_episode=10,
    )

    assert env.action_space.n == len(pokemon_blue_module.ACTION_MAP)
    assert env.observation_space.shape == (1, 144, 160)
    assert env.observation_space.dtype == np.uint8


def test_take_action_calls_pyboy_button(pokemon_blue_module):
    env = pokemon_blue_module.PokemonBlueEnv(rom_path="dummy.gbc")

    env._take_action(0)
    assert env.pyboy.last_button == pokemon_blue_module.ACTION_MAP[0]


def test_get_screen_returns_expected_shape_and_dtype(pokemon_blue_module):
    env = pokemon_blue_module.PokemonBlueEnv(rom_path="dummy.gbc")

    # Give a non-trivial RGB image to validate grayscale conversion.
    rgb = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
    env.pyboy.screen.image = Image.fromarray(rgb, mode="RGB")

    obs = env._get_screen()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (1, 144, 160)
    assert obs.dtype == np.uint8


def test_compute_reward_exploration_only(pokemon_blue_module):
    env = pokemon_blue_module.PokemonBlueEnv(rom_path="dummy.gbc")

    env.pyboy.memory[0xD35E] = 1
    env.pyboy.memory[0xD361] = 2
    env.pyboy.memory[0xD362] = 3

    assert env._compute_reward() == 1.0
    assert env._compute_reward() == 0

    env.pyboy.memory[0xD362] = 4
    assert env._compute_reward() == 1.0


def test_step_ticks_and_terminates_after_steps_per_episode(pokemon_blue_module):
    env = pokemon_blue_module.PokemonBlueEnv(rom_path="dummy.gbc", steps_per_episode=1)

    obs1, reward1, terminated1, truncated1, info1 = env.step(0)
    assert obs1.shape == (1, 144, 160)
    assert reward1 in (0, 1.0)
    assert terminated1 is False
    assert truncated1 is False
    assert isinstance(info1, dict)

    obs2, reward2, terminated2, truncated2, info2 = env.step(0)
    assert obs2.shape == (1, 144, 160)
    assert reward2 in (0, 1.0)
    assert terminated2 is True
    assert truncated2 is False
    assert isinstance(info2, dict)

    # Two tick calls per step
    assert env.pyboy.tick_calls.count((24, False, False)) == 2
    assert env.pyboy.tick_calls.count((1, True, False)) == 2


def test_reset_reinitializes_pyboy_and_clears_episode_state(pokemon_blue_module, tmp_path):
    state_file = tmp_path / "game_start.state"
    state_file.write_bytes(b"state")

    env = pokemon_blue_module.PokemonBlueEnv(
        rom_path="dummy.gbc",
        state_file=str(state_file),
        steps_per_episode=10,
    )

    first_pyboy = env.pyboy
    assert first_pyboy.load_state_calls == 1

    # Mutate episode state
    env.step(0)
    assert env.steps == 1
    assert len(env.explore_map) >= 1

    obs, info = env.reset()
    assert obs.shape == (1, 144, 160)
    assert info == {}

    # Old instance stopped, new instance created
    assert first_pyboy.stop_calls == [False]
    assert env.pyboy is not first_pyboy

    # Episode state reset
    assert env.steps == 0
    assert env.explore_map == {}

    # Saved state loaded again
    assert env.pyboy.load_state_calls == 1
