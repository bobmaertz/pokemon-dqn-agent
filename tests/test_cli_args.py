import sys

import pokemon_blue_agent


def test_parse_args_accepts_episodes_and_steps_aliases(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pokemon_blue_agent.py",
            "--rom_path",
            "POKEMONR.GBC",
            "--state_file",
            "./env_state/pokedex.state",
            "--steps",
            "100",
            "--episodes",
            "3",
            "--episode_log_dir",
            "./episode_logs",
        ],
    )

    args = pokemon_blue_agent.parse_args()

    assert args.rom_path == "POKEMONR.GBC"
    assert args.steps == 100
    assert args.steps_per_episode is None
    assert args.num_episodes == 3
    assert args.episode_log_dir == "./episode_logs"
