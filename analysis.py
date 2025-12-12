import argparse
import re
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt


_LINE_RE = re.compile(
    r"\bstep=(?P<step>-?\d+)\s+"
    r"action=(?P<action>-?\d+)\s+"
    r"reward=(?P<reward>-?\d+(?:\.\d+)?)\s+"
    r"map=(?P<map>-?\d+)\s+"
    r"x=(?P<x>-?\d+)\s+"
    r"y=(?P<y>-?\d+)\s+"
    r"player=(?P<player>\S+)\b"
)


def _iter_log_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.glob("*.log") if p.is_file())
    return [path]


def load_coords_from_logs(path: str) -> list[tuple[int, int, int]]:
    p = Path(path)
    files = _iter_log_files(p)
    if not files:
        raise FileNotFoundError(f"No .log files found under: {p}")

    coords_list: list[tuple[int, int, int]] = []
    for f in files:
        for line_str in f.read_text(encoding="utf-8").splitlines():
            m = _LINE_RE.search(line_str)
            if not m:
                continue
            coords_list.append((int(m.group("map")), int(m.group("x")), int(m.group("y"))))
    return coords_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize movement from episode step logs")
    parser.add_argument(
        "--log_path",
        type=str,
        required=True,
        help="Path to an episode .log file or a directory containing episode_*.log",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coords = load_coords_from_logs(args.log_path)

    # Extract x and y for plotting
    x = [c[1] for c in coords]
    y = [c[2] for c in coords]

    if not x or not y:
        raise ValueError("x and y must not be empty")

    fig, ax = plt.subplots()
    ax.set_title("Agent Movement Over Time")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(min(x) - 2, max(x) + 2)
    ax.set_ylim(min(y) - 2, max(y) + 2)
    ax.grid(True)

    line, = ax.plot([], [], 'bo-', lw=2)
    point, = ax.plot([], [], 'ro', markersize=8)

    # Add a text annotation for the step index on the right side
    step_text = ax.text(
        0.98,
        0.95,
        '',
        transform=ax.transAxes,
        fontsize=12,
        color='purple',
        ha='right',
        va='top')

    def init():
        line.set_data([], [])
        point.set_data([], [])
        step_text.set_text('')
        return line, point, step_text

    def update(frame):
        # Ensure frame does not exceed the length of x or y
        if frame >= len(x):
            frame = len(x) - 1
        line.set_data(x[:frame + 1], y[:frame + 1])
        point.set_data([x[frame]], [y[frame]])  # Wrap in list to ensure sequence
        # Show step and map/frame number
        map_num = coords[frame][0] if coords and frame < len(coords) else 'N/A'
        step_text.set_text(f"Step: {frame}\nMap Num: {map_num}")
        return line, point, step_text

    _ = animation.FuncAnimation(
        fig, update, frames=len(x), init_func=init, blit=True, interval=1, repeat=False
    )

    plt.show()


if __name__ == "__main__":
    main()
