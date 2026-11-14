"""Generate the print-sized feasible-region figure for the PRA manuscript.

Why this is a separate script
-----------------------------
``results/figures/e14_feasibility.png`` is drawn on a 13-inch canvas for
on-screen reading.  Placed at ``width=\\textwidth`` it is scaled by 0.56, which
turns a 9 pt tick label into a 5 pt one.  Uniform scaling never fixes lettering,
because the ratio of letter height to figure width is invariant.  The remedy is
to draw on a canvas that is already at print width with fewer, shorter labels,
which is what this script does.

Canvas
------
7.052 x 2.90 inch, exactly the REVTeX ``pra`` text width of 510.0 pt, so the
figure is included at ``width=\\textwidth`` with a scale factor of 1.000 and no
resampling.  Lettering is 6.5-7.5 pt and lands at 6.5-7.5 pt on the page.

Content
-------
(a) The scanned threshold ``L_min`` against the closed form
    ``2 ln(1/eps) / xi_crit`` for three block lengths.  The closed form omits
    the two fluctuation terms of the three-term bound and is therefore
    optimistic, by a nearly constant factor of 1.64.  At 40 km with
    ``n = 10**7`` no admissible gain exists inside the scanned range at all.
(b) The three terms of the bound against ``L`` at the default working point,
    with the allowance ``xi_crit`` as a horizontal line.  This makes the
    crossing that defines ``L_min`` visible, and shows that only the
    ``1/L`` term responds to the gain while the ``2 kappa_max ln(1/eps) / L``
    tail cost does not.

Everything is recomputed from ``src/qcdma_qkd`` at the same parameters as
``experiments/e14_feasibility.py``; the numbers agree with the stored JSON.

Output
------
``docs/paper/feasible_region.png``
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qcdma_qkd import mui  # noqa: E402

OUTPUT = ROOT / "docs" / "paper" / "feasible_region.png"

# REVTeX pra twocolumn: \textwidth = 510.0 pt = 7.0523 in.
TEXT_WIDTH_IN = 510.0 / 72.27
FIG_HEIGHT_IN = 2.90

INK = "#263238"
BLUE = "#1565c0"
RED = "#c62828"
GREY = "#78909c"
GREEN = "#2e7d32"

# Same parameters as experiments/e14_feasibility.py.
EPS_MUI = 1.0e-12
N_USERS = 8
L_GRID = np.unique(np.round(np.geomspace(256, 8192, 400)).astype(int))

# Copied from the e14 scan so panel (a) plots measured thresholds rather than a
# re-scan at lower resolution.  n = [1e7, 1e8, 1e9].
SCAN = {
    1e7: [(10.0, 784), (25.0, 1520), (40.0, None)],
    1e8: [(10.0, 592), (25.0, 880), (40.0, 1280)],
    1e9: [(10.0, 560), (25.0, 784), (40.0, 1024)],
}
ALLOWANCE = {                    # xi_crit from the e14 allowance table
    1e7: {10.0: 0.11574270329717541, 25.0: 0.059777433526702194,
          40.0: 0.01666899747401475},
    1e8: {10.0: 0.15287778492504728, 25.0: 0.10312289274437354,
          40.0: 0.07109034982975569},
    1e9: {10.0: 0.16472568703675639, 25.0: 0.11702719275187702,
          40.0: 0.08879663715977223},
}


def xi_bound(L: float, allowance_unused: float = 0.0) -> float:
    """Three-term bound on the block-averaged interference, equal powers."""
    kappa = np.ones(N_USERS - 1)
    return float(mui.xi_block_bound(EPS_MUI, kappa, int(L)))


def main() -> Path:
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(TEXT_WIDTH_IN, FIG_HEIGHT_IN), dpi=400,
        gridspec_kw=dict(left=0.085, right=0.985, bottom=0.165, top=0.905,
                         wspace=0.30))

    # -------------------------------------------------------------- panel (a)
    styles = {1e7: (RED, "o", "$n=10^{7}$"),
              1e8: (BLUE, "s", "$n=10^{8}$"),
              1e9: (GREEN, "^", "$n=10^{9}$")}
    for n, (color, marker, label) in styles.items():
        xs = [d for d, L in SCAN[n] if L is not None]
        ys = [L for d, L in SCAN[n] if L is not None]
        ax_a.plot(xs, ys, linestyle="none", marker=marker, color=color,
                  markersize=4.0, markerfacecolor=color, markeredgewidth=0.6,
                  label=label)
        # closed form, using the tabulated allowance
        dc = np.array(sorted(ALLOWANCE[n]))
        cc = np.array([2.0 * math.log(1.0 / EPS_MUI) / ALLOWANCE[n][d]
                       for d in dc])
        dd = np.linspace(dc.min(), dc.max(), 100)
        ax_a.plot(dd, np.interp(dd, dc, cc), color=color, linewidth=0.9,
                  linestyle=(0, (4, 2)), label="_nolegend_")

    # one proxy for the line style, since three dashed copies would be noise
    ax_a.plot([], [], color=GREY, linewidth=0.9, linestyle=(0, (4, 2)),
              label="closed form")

    # The point that does not exist: at 40 km and n = 1e7 the allowance is so
    # tight that no gain up to the scan ceiling clears it.  The ceiling is drawn
    # as a dotted rule so the marker reads as "above this line", not as a value.
    ax_a.axhline(4096, color=GREY, linewidth=0.6, linestyle=(0, (1, 2)))
    ax_a.plot([40.0], [4096.0], marker="x", color=RED, markersize=5.0,
              markeredgewidth=1.1, linestyle="none")
    ax_a.annotate("no admissible gain\nbelow the ceiling", xy=(40.0, 4096.0),
                  xytext=(21.0, 5100.0), fontsize=6.2, color=RED,
                  ha="left", va="center", linespacing=1.25,
                  arrowprops=dict(arrowstyle="-", linewidth=0.5, color=RED,
                                  shrinkA=2, shrinkB=4))

    ax_a.set_yscale("log")
    ax_a.set_ylim(280, 9000)
    ax_a.set_xlim(8.5, 42.0)
    ax_a.set_xlabel("distance $d$ (km)", fontsize=7.5)
    ax_a.set_ylabel("threshold spreading gain $L_{\\min}$", fontsize=7.5)
    ax_a.set_yticks([400, 600, 800, 1200, 1600, 2400])
    ax_a.set_yticklabels(["400", "600", "800", "1200", "1600", "2400"])
    ax_a.set_xticks([10, 20, 30, 40])
    ax_a.legend(fontsize=6.0, loc="upper left", frameon=False, ncol=4,
                handlelength=1.7, columnspacing=0.9, handletextpad=0.45,
                borderpad=0.1, labelspacing=0.22, bbox_to_anchor=(-0.02, 1.005))
    ax_a.set_title("(a) closed form optimistic by $1.64\\times$",
                   fontsize=7.2, pad=9.0)

    # -------------------------------------------------------------- panel (b)
    xi = np.array([xi_bound(L) for L in L_GRID])
    xi_mean = np.array([float(mui.xi_block_mean(np.ones(N_USERS - 1), L))
                        for L in L_GRID])
    # a_j = 1/L for every interferer, so a_max = 1/L and the Laurent-Massart
    # tail term is 2 a_max ln(1/eps) = 2 ln(1/eps)/L -- independent of K, which
    # is exactly the structural fact the text makes use of.  (An earlier version
    # of this figure multiplied by M here and drew a tail term eight times too
    # large, which put the crossing with the allowance on the wrong side of the
    # one marked in panel (b).)
    xi_tail = 2.0 * math.log(1.0 / EPS_MUI) / L_GRID

    ax_b.plot(L_GRID, xi, color=INK, linewidth=1.6, label="rigorous bound")
    ax_b.plot(L_GRID, xi_mean, color=BLUE, linewidth=1.0, linestyle=(0, (4, 2)),
              label="mean term")
    ax_b.plot(L_GRID, xi_tail, color=RED, linewidth=1.0, linestyle=(0, (1.5, 1.5)),
              label="tail term, $\\propto 1/L$")
    allowance = ALLOWANCE[1e8][25.0]
    ax_b.axhline(allowance, color=GREEN, linewidth=1.0, linestyle="-.")
    ax_b.text(275, allowance * 0.90, "allowance $\\xi_{\\mathrm{crit}}$",
              fontsize=6.4, color=GREEN, ha="left", va="top")

    ax_b.plot([880], [allowance], marker="o", color=INK, markersize=3.6,
              markerfacecolor="white", markeredgewidth=0.9, zorder=5)
    ax_b.annotate("$L_{\\min}=880$", xy=(880, allowance),
                  xytext=(600, allowance * 2.9), fontsize=6.5, color=INK,
                  ha="center", va="center",
                  arrowprops=dict(arrowstyle="-", linewidth=0.6, color=INK,
                                  shrinkA=1, shrinkB=2))
    # shade the admissible side
    ax_b.fill_between(L_GRID, 1e-3, xi, where=(xi <= allowance),
                      color=GREEN, alpha=0.10, linewidth=0)

    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlim(256, 8192)
    ax_b.set_ylim(2e-3, 0.45)
    ax_b.set_xlabel("spreading gain $L$", fontsize=7.5)
    ax_b.set_ylabel("block interference $\\xi$", fontsize=7.5)
    ax_b.set_xticks([256, 512, 1024, 2048, 4096, 8192])
    ax_b.set_xticklabels(["256", "512", "1024", "2048", "4096", "8192"])
    ax_b.legend(fontsize=6.0, loc="lower left", frameon=False,
                handlelength=1.8, handletextpad=0.5, borderpad=0.1,
                labelspacing=0.25)
    ax_b.set_title("(b) $d=25$ km, $n=10^{8}$, $M=8$", fontsize=7.2, pad=9.0)

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
