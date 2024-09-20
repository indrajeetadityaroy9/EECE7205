import matplotlib.pyplot as plt
import numpy as np

def plot_node(x, y, text, ax, color):
    ax.plot(x, y, 'o', color=color, markersize=10)
    ax.text(x, y + 0.3, text, fontsize=10, ha='center', va='bottom')

def plot_edge(x1, y1, x2, y2, ax):
    ax.plot([x1, x2], [y1, y2], 'k-', lw=1)

def plot_recursion_tree(x, y, n, level, max_level, ax, colors):
    if n < 1 or level >= max_level:
        return

    plot_node(x, y, f"n/{2**level}", ax, colors[level])
    x_offsets = np.linspace(-50, 50, 8)
    y_offset = -3.5

    for i, x_offset in enumerate(x_offsets):
        child_x = x + x_offset
        child_y = y + y_offset

        plot_edge(x, y, child_x, child_y, ax)
        plot_recursion_tree(child_x, child_y, n // 2, level + 1, max_level, ax, colors)

fig, ax = plt.subplots(figsize=(40, 14))
root_y = 5
ax.set_xlim(-120, 120)
ax.set_ylim(-25, 10)
colors = ['blue', 'green', 'orange', 'purple']
max_level = 3
plot_recursion_tree(0, root_y, 8, 0, max_level, ax, colors)

ax.axis('off')
plt.show()
