import argparse

import numpy as np

from src.agents.DQN import DeepQLearningAgent, Transition
from src.env.pokemon_blue import PokemonBlueEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pokemon Blue DQN smoke runner")
    parser.add_argument(
        "--rom_path",
        type=str,
        required=True,
        help="Path to a legally obtained Gen 1 Pokemon ROM (e.g. POKEMONR.GBC)",
    )
    parser.add_argument(
        "--state_file",
        type=str,
        default=None,
        help="Optional PyBoy save-state file to load at reset",
    )
    parser.add_argument("--render_mode", type=str, default="null")
    parser.add_argument("--emulation_speed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50, help="Number of env steps to run")
    parser.add_argument(
        "--steps_per_episode",
        type=int,
        default=50,
        help="Max steps before env terminates",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = PokemonBlueEnv(
        rom_path=args.rom_path,
        state_file=args.state_file,
        render_mode=args.render_mode,
        emulation_speed=args.emulation_speed,
        steps_per_episode=args.steps_per_episode,
    )

    obs, _ = env.reset()
    agent = DeepQLearningAgent(state_size=obs.shape, action_size=env.action_space.n)

    print("Initial game state:")
    # Use env's info extractor once so we can print player/character data.
    initial_info = env.get_game_state()
    print(initial_info)

    total_reward = 0.0
    for step_idx in range(args.steps):
        action = agent.act(obs)
        next_obs, reward, terminated, _truncated, info = env.step(action)

        total_reward += float(reward)
        print(
            f"step={step_idx:04d} action={action} reward={reward} "
            f"map={info.get('map_num')} x={info.get('x')} y={info.get('y')} "
            f"player={info.get('player_name')}"
        )

        agent.update_memory(
            Transition(
                state=np.asarray(obs),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_obs),
                done=bool(terminated),
            )
        )

        obs = next_obs
        if terminated:
            break

    print({"steps_ran": step_idx + 1, "total_reward": total_reward})
    env.close()


if __name__ == "__main__":
    main()
