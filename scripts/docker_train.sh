#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/docker_train.sh [--build] [--tag IMAGE_TAG] [--gpus auto|all|none] [--cpus N] \
    --rom /path/to/POKEMONR.GBC [--state /path/to/file.state] \
    [--episodes N] [--steps-per-episode N] [--emulation-speed N] \
    [--episode-log-dir DIR] [--wandb-entity X] [--wandb-project Y] [-- EXTRA_ARGS...]

Notes:
  - ROM is bind-mounted read-only into /workspace/rom.gbc
  - State file (if provided) is bind-mounted read-only into /workspace/state.state
  - Episode logs (if enabled) are bind-mounted into /workspace/episode_logs
  - env_state/ is bind-mounted into /workspace/env_state (for reading additional .state files)

Examples:
  scripts/docker_train.sh --build --gpus auto \
    --rom ./POKEMONR.GBC \
    --state ./env_state/pokedex.state \
    --episodes 50 \
    --steps-per-episode 10000 \
    --episode-log-dir ./episode_logs \
    --wandb-entity my-team --wandb-project PokemonRed
EOF
}

repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

abs_path() {
  # macOS-compatible realpath
  python3 - <<'PY' "$1"
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
}

want_build=0
image_tag="pokemon_blue:latest"
gpus="auto"
cpus=""
rom_host=""
state_host=""
episodes=""
steps_per_episode=""
emulation_speed=""
episode_log_dir_host=""
wandb_entity=""
wandb_project=""

extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --build)
      want_build=1
      shift
      ;;
    --tag)
      image_tag="$2"
      shift 2
      ;;
    --gpus)
      gpus="$2"
      shift 2
      ;;
    --cpus)
      cpus="$2"
      shift 2
      ;;
    --rom)
      rom_host="$2"
      shift 2
      ;;
    --state)
      state_host="$2"
      shift 2
      ;;
    --episodes)
      episodes="$2"
      shift 2
      ;;
    --steps-per-episode)
      steps_per_episode="$2"
      shift 2
      ;;
    --emulation-speed)
      emulation_speed="$2"
      shift 2
      ;;
    --episode-log-dir)
      episode_log_dir_host="$2"
      shift 2
      ;;
    --wandb-entity)
      wandb_entity="$2"
      shift 2
      ;;
    --wandb-project)
      wandb_project="$2"
      shift 2
      ;;
    --)
      shift
      extra_args+=("$@")
      break
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$rom_host" ]]; then
  echo "ERROR: --rom is required" >&2
  usage >&2
  exit 2
fi

root="$(repo_root)"

rom_host_abs="$(abs_path "$rom_host")"
if [[ ! -f "$rom_host_abs" ]]; then
  echo "ERROR: ROM not found: $rom_host_abs" >&2
  exit 2
fi

state_host_abs=""
if [[ -n "$state_host" ]]; then
  state_host_abs="$(abs_path "$state_host")"
  if [[ ! -f "$state_host_abs" ]]; then
    echo "ERROR: State file not found: $state_host_abs" >&2
    exit 2
  fi
fi

# Bind-mount optional dirs if present.
env_state_dir="$root/env_state"
episode_logs_dir=""
if [[ -n "$episode_log_dir_host" ]]; then
  episode_logs_dir="$(abs_path "$episode_log_dir_host")"
  mkdir -p "$episode_logs_dir"
fi

# GPU flag handling: only meaningful on Linux + NVIDIA container toolkit.
docker_gpu_args=()
case "$gpus" in
  none)
    ;;
  all)
    docker_gpu_args+=("--gpus" "all")
    ;;
  auto)
    if command -v nvidia-smi >/dev/null 2>&1; then
      # Only add if the docker client supports the flag.
      if docker run --help 2>/dev/null | grep -q -- '--gpus'; then
        docker_gpu_args+=("--gpus" "all")
      fi
    fi
    ;;
  *)
    echo "ERROR: --gpus must be one of: auto|all|none" >&2
    exit 2
    ;;
esac

docker_cpu_args=()
if [[ -n "$cpus" ]]; then
  docker_cpu_args+=("--cpus" "$cpus")
fi

if [[ "$want_build" -eq 1 ]]; then
  (cd "$root" && docker build -t "$image_tag" .)
fi

run_args=()
run_args+=("--rom_path" "/workspace/rom.gbc")
if [[ -n "$state_host_abs" ]]; then
  run_args+=("--state_file" "/workspace/state.state")
fi
if [[ -n "$episodes" ]]; then
  run_args+=("--episodes" "$episodes")
fi
if [[ -n "$steps_per_episode" ]]; then
  run_args+=("--steps_per_episode" "$steps_per_episode")
fi
if [[ -n "$emulation_speed" ]]; then
  run_args+=("--emulation_speed" "$emulation_speed")
fi
if [[ -n "$episode_logs_dir" ]]; then
  run_args+=("--episode_log_dir" "/workspace/episode_logs")
fi
if [[ -n "$wandb_entity" ]]; then
  run_args+=("--wandb_entity" "$wandb_entity")
fi
if [[ -n "$wandb_project" ]]; then
  run_args+=("--wandb_project" "$wandb_project")
fi

# Pass through WANDB_API_KEY if set (common pattern).
docker_env=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  docker_env+=("-e" "WANDB_API_KEY=$WANDB_API_KEY")
fi

mounts=(
  "-v" "$rom_host_abs:/workspace/rom.gbc:ro"
)
if [[ -n "$state_host_abs" ]]; then
  mounts+=("-v" "$state_host_abs:/workspace/state.state:ro")
fi
if [[ -d "$env_state_dir" ]]; then
  mounts+=("-v" "$env_state_dir:/workspace/env_state")
fi
if [[ -n "$episode_logs_dir" ]]; then
  mounts+=("-v" "$episode_logs_dir:/workspace/episode_logs")
fi

set -x
exec docker run --rm -it --init \
  "${docker_cpu_args[@]}" \
  "${docker_gpu_args[@]}" \
  "${docker_env[@]}" \
  "${mounts[@]}" \
  "$image_tag" \
  "${run_args[@]}" \
  "${extra_args[@]}"
