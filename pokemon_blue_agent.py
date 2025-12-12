import argparse
import datetime as _dt
from pathlib import Path

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
        default=None,
        help="Max steps before env terminates (defaults to --steps)",
    )
    parser.add_argument(
        "--episode_log_dir",
        type=str,
        default=None,
        help="Optional directory to write per-episode step logs (one file per episode)",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Optional run identifier used to namespace logs (defaults to a timestamp)",
    )
    return parser.parse_args()


def _format_step_line(step_idx: int, action: int, reward: float, info: dict) -> str:
    return (
        f"step={step_idx} action={action} reward={reward} "
        f"map={info.get('map_num')} x={info.get('x')} y={info.get('y')} "
        f"player={info.get('player_name')}"
    )


def _open_episode_log(episode_log_dir: str | None, run_id: str, episode_idx: int):
    if not episode_log_dir:
        return None
    run_dir = Path(episode_log_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"episode_{episode_idx:04d}.log"
    # Line-buffered so logs are readable during long runs.
    return path.open("w", encoding="utf-8", buffering=1)


def main() -> None:
    args = parse_args()

    steps_per_episode = args.steps if args.steps_per_episode is None else args.steps_per_episode
    run_id = args.run_id or _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    quiet = args.episode_log_dir is not None

    env = PokemonBlueEnv(
        rom_path=args.rom_path,
        state_file=args.state_file,
        render_mode=args.render_mode,
        emulation_speed=args.emulation_speed,
        steps_per_episode=steps_per_episode,
    )

    obs, _ = env.reset()
    agent = DeepQLearningAgent(state_size=obs.shape, action_size=env.action_space.n)

    initial_info = env.get_game_state()
    if not quiet:
        print("Initial game state:")
        print(initial_info)

    total_reward = 0.0
    global_step_idx = 0
    episode_idx = 0

    while global_step_idx < args.steps:
        if episode_idx > 0:
            obs, _ = env.reset()

        episode_log = _open_episode_log(args.episode_log_dir, run_id, episode_idx)
        try:
            if episode_log is not None:
                episode_log.write(f"# run_id={run_id} episode={episode_idx}\n")
                if episode_idx == 0:
                    episode_log.write(f"# initial_info={initial_info}\n")
            while global_step_idx < args.steps:
                action = agent.act(obs)
                next_obs, reward, terminated, _truncated, info = env.step(action)

                total_reward += float(reward)
                line = _format_step_line(global_step_idx, int(action), float(reward), info)
                if episode_log is not None:
                    episode_log.write(line + "\n")
                elif not quiet:
                    print(line)

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
                global_step_idx += 1
                if terminated:
                    break
        finally:
            if episode_log is not None and global_step_idx >= args.steps:
                episode_log.write(f"# summary={{'steps_ran': {global_step_idx}, 'total_reward': {total_reward}}}\n")
            if episode_log is not None:
                episode_log.close()

        episode_idx += 1
        if not terminated:
            break

    if not quiet:
        print({"steps_ran": global_step_idx, "total_reward": total_reward})
    env.close()


if __name__ == "__main__":
    main()
