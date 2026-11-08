"""Shared matplotlib helpers for all experiments.

Uses a non-interactive backend and a consistent style so figures are
reproducible when generated head-less from the terminal.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")          # must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[2] / "results" / "figures"
DATA_DIR = Path(__file__).resolve().parents[2] / "results" / "data"


def setup() -> None:
    """Apply the shared style and make sure output directories exist."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "figure.autolayout": True,
    })


def save(fig, name: str, close: bool = True) -> Path:
    """Save a figure into ``results/figures`` and return its path."""
    setup()
    path = FIG_DIR / (name if name.endswith((".pdf", ".png")) else f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    if close:
        plt.close(fig)
    print(f"  figure -> {path.relative_to(path.parents[2])}")
    return path


def annotate(ax, text: str, xy=(0.03, 0.97), **kw):
    """Top-left annotation box."""
    kw.setdefault("fontsize", 8)
    kw.setdefault("va", "top")
    kw.setdefault("ha", "left")
    ax.text(*xy, text, transform=ax.transAxes, **kw)
