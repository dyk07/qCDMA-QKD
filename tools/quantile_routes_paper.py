"""Generate the print-sized quantile-route figure for the PRA manuscript.

Why this is a separate script
-----------------------------
``results/figures/e10_model_vs_bound.png`` is a 13-inch on-screen figure.  At
``width=\\textwidth`` a 9 pt label inside it shrinks to 5 pt, because uniform
scaling preserves the ratio of letter height to figure width.  The fix is to
draw on a canvas that is already at print width with fewer, shorter labels.

Canvas
------
7.052 x 2.55 inch, exactly the REVTeX ``pra`` text width of 510.0 pt, so the
figure is placed at scale 1.000.

Content
-------
The two panels show the same four excess-noise levels from two directions.

(a) Absolute levels: the exact quantile, the Gaussian-matched approximation and
    the Laurent--Massart bound, against ``eps``.  The Gaussian route lies below
    the exact level at every ``eps`` and the deficit grows as ``eps`` falls,
    while the bound stays above it by a shrinking margin.  A moment-matched
    model is therefore unsafe in the direction that matters, and becomes more
    unsafe exactly where the security parameter is tightened.

(b) Relative deviation from the exact value, which is what the manuscript quotes
    as $-39\\%$ to $-57\\%$ for the Gaussian route and $+34\\%$ to $+25\\%$ for
    the bound.

Every number is read from ``results/data/e10_model_vs_bound.json`` so the figure
cannot drift from the stored experiment.

Output
------
``docs/paper/quantile_routes.png``
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "data" / "e10_model_vs_bound.json"
OUTPUT = ROOT / "docs" / "paper" / "quantile_routes.png"

TEXT_WIDTH_IN = 510.0 / 72.27
FIG_HEIGHT_IN = 2.55

INK = "#263238"
BLUE = "#1565c0"
RED = "#c62828"
GREEN = "#2e7d32"
GREY = "#78909c"


def main() -> Path:
    rows = json.loads(DATA.read_text())["mui"]
    eps = np.array([r["eps_mui"] for r in rows])
    exact = np.array([r["xi_exact"] for r in rows])
    gauss = np.array([r["xi_gaussian"] for r in rows])
    bound = np.array([r["xi_bound"] for r in rows])

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(TEXT_WIDTH_IN, FIG_HEIGHT_IN), dpi=400,
        gridspec_kw=dict(left=0.088, right=0.978, bottom=0.175, top=0.895,
                         wspace=0.30))

    # -------------------------------------------------------------- panel (a)
    ax_a.plot(eps, exact, marker="o", color=INK, markersize=3.8,
              linewidth=1.2, markerfacecolor="white", markeredgewidth=1.0,
              label="exact quantile")
    ax_a.plot(eps, gauss, marker="v", color=RED, markersize=3.8,
              linewidth=1.0, linestyle=(0, (4, 2)), label="Gaussian-matched")
    ax_a.plot(eps, bound, marker="^", color=BLUE, markersize=3.8,
              linewidth=1.0, linestyle=(0, (1.5, 1.5)), label="rigorous bound")

    # shade the unsafe side of the exact curve
    ax_a.fill_between(eps, 0.0, exact, color=RED, alpha=0.055, linewidth=0)
    ax_a.text(3.0e-9, 0.0165, "model side\n(under-provisioned)", fontsize=6.0,
              color=RED, ha="left", va="center", linespacing=1.25, alpha=0.95)

    ax_a.set_xscale("log")
    ax_a.set_xlim(5e-7, 2.5e-16)
    ax_a.set_ylim(0.0, 0.125)
    ax_a.set_xticks([1e-6, 1e-9, 1e-12, 1e-15])
    ax_a.set_xticklabels(["$10^{-6}$", "$10^{-9}$", "$10^{-12}$", "$10^{-15}$"])
    ax_a.set_xlabel("interference failure probability $\\varepsilon_{\\mathrm{MUI}}$",
                    fontsize=7.5)
    ax_a.set_ylabel("block interference $\\xi$", fontsize=7.5)
    ax_a.legend(fontsize=6.0, loc="upper left", frameon=False,
                handlelength=1.8, handletextpad=0.45, borderpad=0.1,
                labelspacing=0.25)
    ax_a.set_title("(a) absolute levels", fontsize=7.2, pad=8.0)

    # -------------------------------------------------------------- panel (b)
    g_rel = 100.0 * (gauss / exact - 1.0)
    b_rel = 100.0 * (bound / exact - 1.0)
    ax_b.axhline(0.0, color=INK, linewidth=0.8)
    ax_b.plot(eps, g_rel, marker="v", color=RED, markersize=3.8,
              linewidth=1.1, linestyle=(0, (4, 2)), label="Gaussian-matched")
    ax_b.plot(eps, b_rel, marker="^", color=BLUE, markersize=3.8,
              linewidth=1.1, linestyle=(0, (1.5, 1.5)), label="rigorous bound")
    ax_b.fill_between(eps, 0.0, g_rel, color=RED, alpha=0.075, linewidth=0)

    # endpoint percentages, each pushed into clear space beside its marker
    for xs, ys, color, offset, ha, va in (
        (eps[0], g_rel[0], RED, (5.0, -5.0), "left", "top"),
        (eps[0], b_rel[0], BLUE, (5.0, 5.0), "left", "bottom"),
        (eps[-1], g_rel[-1], RED, (-7.0, -6.0), "right", "top"),
        (eps[-1], b_rel[-1], BLUE, (-7.0, 6.0), "right", "bottom"),
    ):
        ax_b.annotate(f"{ys:+.0f}%", xy=(xs, ys), xytext=offset,
                      textcoords="offset points", fontsize=6.2, color=color,
                      ha=ha, va=va)

    ax_b.set_xscale("log")
    ax_b.set_xlim(5e-7, 2.5e-16)
    ax_b.set_ylim(-72.0, 46.0)
    ax_b.set_xticks([1e-6, 1e-9, 1e-12, 1e-15])
    ax_b.set_xticklabels(["$10^{-6}$", "$10^{-9}$", "$10^{-12}$", "$10^{-15}$"])
    ax_b.set_xlabel("interference failure probability $\\varepsilon_{\\mathrm{MUI}}$",
                    fontsize=7.5)
    ax_b.set_ylabel("deviation from exact (%)", fontsize=7.5)
    ax_b.legend(fontsize=6.0, loc="lower left", frameon=False,
                handlelength=1.8, handletextpad=0.45, borderpad=0.1,
                labelspacing=0.25)
    ax_b.set_title("(b) relative to the exact quantile", fontsize=7.2, pad=8.0)

    for ax in (ax_a, ax_b):
        ax.tick_params(labelsize=6.6, length=2.4, width=0.6, pad=2.0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(0.6)
        ax.grid(True, which="major", alpha=0.20, linewidth=0.45)
        ax.set_axisbelow(True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, facecolor="white", dpi=400)
    plt.close(fig)
    print(f"figure -> {OUTPUT}")
    return OUTPUT


if __name__ == "__main__":
    main()
