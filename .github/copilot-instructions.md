# Copilot instructions (pokemon_blue_claude_workspace)

## Big picture
- Training entrypoint is `pokemon_blue_agent.py` (CLI + W&B logging) which wires `PokemonBlueEnv` (`src/env/pokemon_blue.py`) to `DeepQLearningAgent` (`src/agents/DQN.py`).
- `PokemonBlueEnv` wraps PyBoy and exposes a Gym-style API: observation is a grayscale frame shaped `(1, 144, 160)` and actions are mapped via `ACTION_MAP`.
- Reward is currently exploration-only: first-time visits to `(map_num, x, y)` from Game Boy RAM addresses `0xD35E/0xD361/0xD362` yield `1.0`.

## How to run (local)
- Requires a legally obtained ROM (default `./POKEMONR.GBC`) and optionally a saved state file (default `./env_state/game_start.state`). These are intentionally ignored by git (`.gitignore`).
- Main training command is:
  - `python pokemon_blue_agent.py --rom_path ./POKEMONR.GBC --state_file ./env_state/game_start.state`
- W&B is optional; if used, it’s configured via CLI args or env vars `WANDB_ENTITY`/`WANDB_PROJECT`.

## How to run (Docker)
- `Dockerfile` builds a CUDA runtime image and runs `pokemon_blue_agent.py` as the entrypoint.
- `Makefile` has `build`/`run` targets; `run` mounts `/workspace/env_state` as a volume for state files.

## Tests and workflows
- Unit tests live under `tests/` and focus on the DQN agent (`tests/test_dqn_agent.py`).
- Run tests with:
  - `pytest -q`
- Formatting is done via autopep8 (also enforced via pre-commit):
  - `autopep8 -i --aggressive *.py`

## Project conventions / patterns
- Source layout is `src/` with modules imported as `src.agents...` and `src.env...` in production code.
- Tests currently tweak `sys.path` to import from `src/` directly (see `tests/test_dqn_agent.py`). Keep this in mind when moving files/renaming modules.
- Agent device selection prefers Apple Silicon (`torch.mps`) then CUDA, else CPU (see `DeepQLearningAgent.__init__`).

## Gotchas (verify before changing)
- `pokemon_blue_agent.py` expects CLI arg `--steps_per_episode`, but the code currently references `args.num_steps_per_episode` when constructing `PokemonBlueEnv`.
- `Makefile`’s `venv` target activates `venv/bin/activate` (repo may also use `.venv/` locally).
- `Readme.md` mentions a different script name; treat `pokemon_blue_agent.py` as the canonical entrypoint.
