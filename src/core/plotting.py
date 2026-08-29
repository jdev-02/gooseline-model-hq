"""Shared plotting configuration.

IEEE-style defaults but with text.usetex off by default since not every
machine has a LaTeX install. Flip USE_TEX = True if you have one.
"""

from pathlib import Path

import matplotlib.pyplot as plt

USE_TEX = False

PLOT_PARAMS = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "text.usetex": USE_TEX,
    "font.family": "serif" if USE_TEX else "sans-serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.figsize": (7, 4),
    "axes.grid": True,
    "grid.alpha": 0.3,
}


def apply_style() -> None:
    plt.rcParams.update(PLOT_PARAMS)


def save_figure(fig, name: str, phase: str) -> Path:
    """Save fig as PDF to figures/{phase}/{name}.pdf and return the path."""
    fig_dir = Path(f"figures/{phase}")
    fig_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = fig_dir / f"{name}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    return pdf_path
