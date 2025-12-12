import argparse
import datetime as _dt
import os
from pathlib import Path

import numpy as np

from src.agents.DQN import DeepQLearningAgent, Transition
from src.env.pokemon_blue import PokemonBlueEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pokemon Blue DQN trainer")
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
    parser.add_argument(
        "--steps",
        type=int,
        default=10000,
        help="Fallback for --steps_per_episode (kept for backward-compat)",
    )
    parser.add_argument(
        "--steps_per_episode",
        type=int,
        default=None,
        help="Max steps before env terminates (defaults to --steps)",
    )
    parser.add_argument(
        "--num_episodes",
        "--episodes",
        dest="num_episodes",
        type=int,
        default=1,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=None,
        help="Optional global step limit across all episodes (defaults to num_episodes * steps_per_episode)",
    )
    parser.add_argument(
        "--train_every",
        type=int,
        default=4,
        help="Train the DQN every N environment steps",
    )
    parser.add_argument(
        "--replay_memory_size",
        type=int,
        default=500,
        help="Replay memory size",
    )
    parser.add_argument(
        "--minibatch_size",
        type=int,
        default=64,
        help="Minibatch size for training",
    )
    parser.add_argument(
        "--epsilon_decay",
        type=float,
        default=0.99,
        help="Epsilon decay applied when training occurs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Optimizer learning rate",
    )
    parser.add_argument(
        "--epsilon_min",
        type=float,
        default=0.01,
        help="Minimum exploration rate",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="pokemon_blue_dqn",
        help="Model name prefix for saving",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY", ""),
        help="Weights & Biases entity (optional)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", ""),
        help="Weights & Biases project (optional)",
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
    total_steps_limit = (
        (steps_per_episode * int(args.num_episodes))
        if args.total_steps is None
        else int(args.total_steps)
    )
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
    agent = DeepQLearningAgent(
        state_size=obs.shape,
        action_size=env.action_space.n,
        replay_memory_size=int(args.replay_memory_size),
        minibatch_size=int(args.minibatch_size),
        epsilon_decay=float(args.epsilon_decay),
        learning_rate=float(args.learning_rate),
        epsilon_min=float(args.epsilon_min),
    )

    initial_info = env.get_game_state()
    if not quiet:
        print("Initial game state:")
        print(initial_info)

    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(
            entity=args.wandb_entity or None,
            project=args.wandb_project,
            config={
                "learning_rate": agent.learning_rate,
                "device": str(agent.device),
                "epsilon": agent.epsilon,
                "epsilon_decay": agent.epsilon_decay,
                "gamma": agent.gamma,
                "minibatch_size": agent.minibatch_size,
                "replay_memory_size": agent.replay_memory_size,
                "architecture": "CNN",
                "episodes": int(args.num_episodes),
                "steps_per_episode": int(steps_per_episode),
                "emulation_speed": int(args.emulation_speed),
                "state_file_name": args.state_file,
                "rom_path": args.rom_path,
            },
        )

    total_reward = 0.0
    global_step_idx = 0

    for episode_idx in range(int(args.num_episodes)):
        if global_step_idx >= total_steps_limit:
            break

        obs, _ = env.reset()
        episode_reward = 0.0
        episode_losses: list[float] = []
        episode_qs: list[float] = []

        episode_log = _open_episode_log(args.episode_log_dir, run_id, episode_idx)
        try:
            if episode_log is not None:
                episode_log.write(f"# run_id={run_id} episode={episode_idx}\n")
                if episode_idx == 0:
                    episode_log.write(f"# initial_info={initial_info}\n")

            terminated = False
            for _step_in_ep in range(int(steps_per_episode)):
                if global_step_idx >= total_steps_limit:
                    break

                action = agent.act(obs)
                next_obs, reward, terminated, _truncated, info = env.step(action)

                reward_f = float(reward)
                total_reward += reward_f
                episode_reward += reward_f

                line = _format_step_line(global_step_idx, int(action), reward_f, info)
                if episode_log is not None:
                    episode_log.write(line + "\n")
                elif not quiet:
                    print(line)

                agent.update_memory(
                    Transition(
                        state=np.asarray(obs),
                        action=int(action),
                        reward=reward_f,
                        next_state=np.asarray(next_obs),
                        done=bool(terminated),
                    )
                )

                if int(args.train_every) > 0 and (global_step_idx % int(args.train_every) == 0):
                    result = agent.train()
                    if result is not None:
                        loss, q = result
                        episode_losses.append(float(loss))
                        episode_qs.append(float(q))
                        if agent.epsilon > agent.epsilon_min:
                            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

                obs = next_obs
                global_step_idx += 1
                if terminated:
                    break
        finally:
            if episode_log is not None:
                avg_loss = (sum(episode_losses) / len(episode_losses)) if episode_losses else 0.0
                avg_q = (sum(episode_qs) / len(episode_qs)) if episode_qs else 0.0
                episode_log.write(
                    "# summary={"
                    f"'episode': {episode_idx}, "
                    f"'steps_ran': {global_step_idx}, "
                    f"'episode_reward': {episode_reward}, "
                    f"'total_reward': {total_reward}, "
                    f"'epsilon': {agent.epsilon}, "
                    f"'average_loss': {avg_loss}, "
                    f"'average_q': {avg_q}"
                    "}\n"
                )
                episode_log.close()

        if wandb_run is not None:
            avg_loss = (sum(episode_losses) / len(episode_losses)) if episode_losses else 0.0
            avg_q = (sum(episode_qs) / len(episode_qs)) if episode_qs else 0.0
            wandb_run.log(
                {
                    "episode": episode_idx,
                    "episode_reward": episode_reward,
                    "total_reward": total_reward,
                    "average_loss": avg_loss,
                    "average_q": avg_q,
                    "num_steps": global_step_idx,
                    "epsilon": agent.epsilon,
                }
            )

        if terminated is False:
            # If we didn't terminate naturally, still proceed to next episode (reset will happen).
            pass

    agent.save(args.model_name)
    if wandb_run is not None:
        wandb_run.finish()

    if not quiet:
        print({"steps_ran": global_step_idx, "total_reward": total_reward})
    env.close()


if __name__ == "__main__":
    main()
