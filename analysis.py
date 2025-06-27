import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Example: Replace this with your actual coordinate data
# Each tuple: (map_num, x, y)
coords = [
    (1, 10, 10), (1, 11, 10), (1, 12, 10), (1, 13, 10), (1, 14, 10),
    (1, 15, 10), (1, 16, 10), (1, 17, 10), (1, 18, 10), (1, 19, 10),
    (1, 20, 10), (1, 21, 10), (1, 22, 10), (1, 23, 10), (1, 24, 10),
    (1, 25, 10), (1, 26, 10), (1, 27, 10), (1, 28, 10), (1, 29, 10),
    (1, 30, 10), (1, 30, 11), (1, 30, 12), (1, 30, 13), (1, 30, 14),
    (1, 30, 15), (1, 30, 16), (1, 30, 17), (1, 30, 18), (1, 30, 19),
    (1, 30, 20), (1, 29, 20), (1, 28, 20), (1, 27, 20), (1, 26, 20),
    (1, 25, 20), (1, 24, 20), (1, 23, 20), (1, 22, 20), (1, 21, 20),
    (1, 20, 20), (1, 19, 20), (1, 18, 20), (1, 17, 20), (1, 16, 20),
    (1, 15, 20), (1, 14, 20), (1, 13, 20), (1, 12, 20), (1, 11, 20),
    (1, 10, 20), (1, 10, 19), (1, 10, 18), (1, 10, 17), (1, 10, 16),
    (1, 10, 15), (1, 10, 14), (1, 10, 13), (1, 10, 12), (1, 10, 11),
    (1, 10, 10), (1, 11, 11), (1, 12, 12), (1, 13, 13), (1, 14, 14),
    (1, 15, 15), (1, 16, 16), (1, 17, 17), (1, 18, 18), (1, 19, 19),
    (1, 20, 20), (1, 21, 19), (1, 22, 18), (1, 23, 17), (1, 24, 16),
    (1, 25, 15), (1, 26, 14), (1, 27, 13), (1, 28, 12), (1, 29, 11),
    (1, 30, 10)
]

# Extract x and y for plotting
x = [c[1] for c in coords]
y = [c[2] for c in coords]

if not x or not y:
    raise ValueError("x and y must not be empty")

fig, ax = plt.subplots()
ax.set_title("Agent Movement Over Time")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_xlim(min(x)-2, max(x)+2)
ax.set_ylim(min(y)-2, max(y)+2)
ax.grid(True)

line, = ax.plot([], [], 'bo-', lw=2)
point, = ax.plot([], [], 'ro', markersize=8)

# Add a text annotation for the step index on the right side
step_text = ax.text(0.98, 0.95, '', transform=ax.transAxes, fontsize=12, color='purple', ha='right', va='top')

def init():
    line.set_data([], [])
    point.set_data([], [])
    step_text.set_text('')
    return line, point, step_text

def update(frame):
    # Ensure frame does not exceed the length of x or y
    if frame >= len(x):
        frame = len(x) - 1
    line.set_data(x[:frame+1], y[:frame+1])
    point.set_data([x[frame]], [y[frame]])  # Wrap in list to ensure sequence
    # Show step and map/frame number
    map_num = coords[frame][0] if coords and frame < len(coords) else 'N/A'
    step_text.set_text(f"Step: {frame}\nMap Num: {map_num}")
    return line, point, step_text

ani = animation.FuncAnimation(
    fig, update, frames=len(x), init_func=init, blit=True, interval=1, repeat=False
)

plt.show()