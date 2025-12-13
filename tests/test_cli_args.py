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
            "--gamma",
            "0.9",
            "--epsilon_start",
            "0.5",
            "--replay_warmup",
            "123",
            "--target_update_every",
            "42",
        ],
    )

    args = pokemon_blue_agent.parse_args()

    assert args.rom_path == "POKEMONR.GBC"
    assert args.steps == 100
    assert args.steps_per_episode is None
    assert args.num_episodes == 3
    assert args.episode_log_dir == "./episode_logs"
    assert args.gamma == 0.9
    assert args.epsilon_start == 0.5
    assert args.replay_warmup == 123
    assert args.target_update_every == 42
