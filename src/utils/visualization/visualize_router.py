import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_alpha_density(alpha_dict: dict[str, list[float]], output_dir: str, filename: str = "alpha_density.png"):
    """Plot density curves of router alpha values for each problem category.

    Args:
        alpha_dict: mapping from category name to list of alpha values,
                    e.g. {"Original": [...], "Simple": [...], "Hard": [...]}.
        output_dir: directory to save the plot.
        filename: output file name.
    """
    colors = {"Original": "#1f77b4", "Simple": "#2ca02c", "Hard": "#d62728"}
    x_grid = np.linspace(0, 1, 500)

    fig, ax = plt.subplots(figsize=(8, 5))

    for label in ("Original", "Simple", "Hard"):
        values = np.array(alpha_dict.get(label, []), dtype=np.float64)
        if len(values) < 2:
            continue
        values = np.clip(values, 1e-6, 1 - 1e-6)
        kde = gaussian_kde(values, bw_method="silverman")
        density = kde(x_grid)
        ax.plot(x_grid, density, color=colors[label], linewidth=2, label=label)
        ax.fill_between(x_grid, density, alpha=0.15, color=colors[label])

    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$\alpha$", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.set_title(r"Router $\alpha$ Distribution by Problem Category", fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Alpha density plot saved to {save_path}")
