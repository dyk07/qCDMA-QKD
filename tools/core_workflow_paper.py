"""Generate the print-sized core-workflow figure for the PRA manuscript.

Why this is a separate script
-----------------------------
``tools/core_workflow.py`` draws the canonical diagram on a 15 x 8 inch canvas
for on-screen reading.  A journal figure is reduced to the width of a
two-column text block, about 6.6 inch, and uniform scaling does **not** change
the ratio of letter height to figure width: a 9-point label on a 15-inch canvas
appears at 9 * 6.6 / 15 = 4 point once the figure is placed at text width,
which is below what a typesetter will accept.  Shrinking the canvas while
keeping the font sizes does not help either, because the boxes shrink and the
labels do not.

The only way to obtain readable lettering is to reduce the ratio of label width
to canvas width, that is, to redraw the workflow on a canvas that is already at
print width with shorter wording inside the boxes.  That is what this script
does.  The content is the same workflow as ``results/figures/core_workflow.png``.

Canvas
------
7.052 x 4.231 inch, which is exactly the REVTeX `pra` text width of 510.0 pt
(``\textwidth``) along with its proportional height, so the figure is included
at ``width=\textwidth`` with a scale factor of 1.000 and no resampling.
Lettering is 5.9-7.4 pt and lands at 5.9-7.4 pt on the page.

Output
------
``docs/paper/core_workflow.png``
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "paper" / "core_workflow.png"

# REVTeX pra twocolumn: \textwidth = 510.0 pt = 7.0523 in.  Drawing on a canvas
# of exactly that width means the PDF is placed at scale 1.000, so a 7 pt label
# prints at 7 pt.
TEXT_WIDTH_IN = 510.0 / 72.27
FIG_HEIGHT_IN = TEXT_WIDTH_IN * 3.96 / 6.6

# Palette shared with the on-screen version.
BLUE = "#e3f2fd"
INDIGO = "#e8eaf6"
TEAL = "#e0f2f1"
GREEN = "#e8f5e9"
PURPLE = "#f3e5f5"
LILAC = "#ede7f6"
RED = "#ffebee"

INK = "#263238"
GREEN_INK = "#2e7d32"


def box(ax, x, y, w, h, text, color, fontsize=7.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=1.1",
        facecolor=color, edgecolor=INK, linewidth=0.8,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color="#17202a", linespacing=1.3)


def arrow(ax, p0, p1, color=INK, lw=0.9, ms=7):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
        color=color, shrinkA=0, shrinkB=0,
    ))


def line(ax, p0, p1, color=INK, lw=0.9):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linewidth=lw,
            solid_capstyle="butt")


def main() -> Path:
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, FIG_HEIGHT_IN), dpi=400)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ---------------------------------------------------------------- parties
    for x0, label in ((1.0, "ALICE"), (90.0, "BOB")):
        ax.add_patch(plt.Rectangle((x0, 8.0), 9.0, 44.0, facecolor="#e8f5e9",
                                   edgecolor=GREEN_INK, linewidth=0.8))
        ax.text(x0 + 4.5, 30.0, label, ha="center", va="center", fontsize=7.0,
                fontweight="bold", color="#1b5e20", rotation=90)

    # ------------------------------------------------------- bootstrap (top)
    box(ax, 13.0, 44.0, 11.0, 6.4, "root $K_0$", BLUE)
    box(ax, 27.0, 44.0, 15.0, 6.4, "split into\n$a_0,\\ c_0$", INDIGO, 6.8)
    box(ax, 45.0, 44.0, 13.0, 6.4, "codebook\n$\\mathrm{PRF}_{c_0}$", TEAL, 6.8)
    arrow(ax, (24.0, 47.2), (27.0, 47.2))
    arrow(ax, (42.0, 47.2), (45.0, 47.2))
    ax.text(13.0, 52.4, "one-time bootstrap", fontsize=6.4, color="#546e7a",
            style="italic", ha="left", va="center")

    # the bootstrap codebook is what the first round expands locally
    line(ax, (51.5, 44.0), (51.5, 41.6))
    line(ax, (51.5, 41.6), (36.0, 41.6))
    arrow(ax, (36.0, 41.6), (36.0, 36.6))

    # ------------------------------------------------------------ main round
    box(ax, 13.0, 30.0, 13.0, 6.6, "random\nbits", BLUE)
    box(ax, 29.0, 30.0, 15.0, 6.6, "CDMA\nencoding", PURPLE)
    box(ax, 61.5, 30.0, 13.5, 6.6, "CDMA\ndecoding", PURPLE)
    box(ax, 77.5, 30.0, 11.5, 6.6, "parameter\nestimation", GREEN, 6.2)

    arrow(ax, (26.0, 33.3), (29.0, 33.3))
    arrow(ax, (44.0, 33.3), (46.0, 33.3))
    arrow(ax, (60.0, 33.3), (61.5, 33.3))
    arrow(ax, (75.0, 33.3), (77.5, 33.3))

    # quantum channel
    for dy in (-1.5, 0.0, 1.5):
        xs = [46.0 + i * 0.5 for i in range(29)]
        ys = [33.3 + dy + 0.42 * (1 if i % 2 else -1) for i in range(29)]
        ax.plot(xs, ys, color="#1565c0", linewidth=0.7)
    ax.text(53.0, 38.0, "multi-user quantum channel", fontsize=6.2,
            color="#1565c0", ha="center", va="center")

    # ------------------------------------------------------- Bob's decisions
    box(ax, 53.0, 18.6, 21.0, 7.0, "reconcile +\nprivacy amplification",
        LILAC, 6.2)
    box(ax, 77.5, 19.0, 11.5, 6.4, "validate", BLUE, 6.6)

    arrow(ax, (83.25, 30.0), (83.25, 25.4))
    arrow(ax, (77.5, 22.1), (74.0, 22.1), color=GREEN_INK)
    ax.text(75.7, 23.3, "yes", fontsize=6.0, color=GREEN_INK, ha="center",
            va="bottom")

    # ------------------------------------------------------------- bottom row
    box(ax, 60.0, 9.0, 15.0, 6.4, "$K_n$", GREEN, 7.4)
    box(ax, 77.5, 9.0, 11.5, 6.4, "abort\nsafely", RED, 6.4)
    box(ax, 20.0, 9.0, 32.0, 6.4,
        "$\\mathrm{KDF}_{K_n}\\rightarrow a_{n+1},\\ c_{n+1},\\ u_n$",
        INDIGO, 6.8)

    arrow(ax, (62.0, 18.6), (62.0, 15.4))
    arrow(ax, (60.0, 12.2), (52.0, 12.2))
    arrow(ax, (83.25, 19.0), (83.25, 15.4), color="#c62828")
    ax.text(84.2, 17.2, "no", fontsize=6.0, color="#c62828", ha="left",
            va="center")

    # feedback into the next round, and the application output
    line(ax, (20.0, 12.2), (16.0, 12.2), color=GREEN_INK)
    arrow(ax, (16.0, 12.2), (16.0, 30.0), color=GREEN_INK)
    ax.text(15.2, 21.0, "next round", fontsize=6.0, color=GREEN_INK,
            rotation=90, ha="center", va="center")

    arrow(ax, (36.0, 9.0), (36.0, 4.9), color=GREEN_INK)
    ax.text(37.0, 6.5, "$u_n$ to the application", fontsize=6.2,
            color=GREEN_INK, ha="left", va="center")

    ax.text(1.0, 1.8,
            "Model boundary: $K_n$ is produced by the finite-size CV-QKD rate; "
            "the loop redistributes it and does not extend the quantum bound.",
            fontsize=5.9, color="#455a64", ha="left", va="center")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white", dpi=400)
    plt.close(fig)
    print(f"figure -> {OUTPUT}")
    return OUTPUT


if __name__ == "__main__":
    main()
