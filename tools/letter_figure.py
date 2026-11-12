"""Build the single figure for the code-secrecy Letter (APS single column).

Everything plotted is read from committed experiment output; no new simulation
runs here, so the Letter cannot drift from the repository's numerical baseline.

Panel (a)  Fraction of codes recovered exactly versus the number of observed
           symbols.  Points with ``n >= 1e4`` come from
           ``results/data/e11_covariance_attack.json``.  The onset region comes
           from ``results/data/e12_refresh_period.json``, which swept three
           independent codebook draws; the shaded band spans the worst draw to
           the mean, showing the transition is statistical rather than sharp.
           The analytic *sufficient* threshold ``n_th`` is overlaid: the measured
           onset lies about two orders of magnitude below it, so the bound is
           safe but loose.

Panel (b)  Cost of the only available defence: recomputing the codebook every
           ``T`` symbols costs ``ML/T`` bits of PRF output per symbol.

Output: ``docs/paper/letter_fig.png``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qcdma_qkd import plotting  # noqa: E402  (sets rcParams)

plt = plotting.plt

DATA = ROOT / "results" / "data"
OUT = ROOT / "docs" / "paper" / "letter_fig.png"

C_EQUAL = "#186a3b"
C_SPREAD = "#1f618d"
C_WEAK = "#a93226"
C_BOUND = "#566573"


def _load(name: str) -> dict:
    return json.loads((DATA / name).read_text())


def onset_series() -> dict:
    rows = _load("e12_refresh_period.json")["measured"]
    n = np.array([float(r["T"]) for r in rows])
    mean = np.array([float(r["recovery_fraction"]) for r in rows])
    low = np.array([float(r["recovery_min"]) for r in rows])
    order = np.argsort(n)
    return dict(n=n[order], mean=mean[order], low=low[order])


def grid_series(e11: dict, key: str) -> dict:
    rows = e11["recovery"][key]
    return dict(n=np.array([float(r["n_samples"]) for r in rows]),
                frac=np.array([float(r["recovery_fraction"]) for r in rows]))


def main() -> None:
    e11 = _load("e11_covariance_attack.json")
    e12 = _load("e12_refresh_period.json")

    on = onset_series()
    equal = grid_series(e11, "equal")
    spread = grid_series(e11, "spread")
    weak = grid_series(e11, "weak")

    n_th = float(e11["n_theory"]["equal"])
    round_symbols = float(e11["total_symbols_per_round"])

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.4, 2.85),
                                 constrained_layout=True,
                                 gridspec_kw=dict(height_ratios=[1.15, 1.0]))

    # ---------------- (a) recovery versus n -------------------------------
    ax.set_xscale("log")
    ax.fill_between(on["n"], on["low"], on["mean"], color=C_EQUAL, alpha=0.18,
                    lw=0, zorder=1)

    n_all = np.concatenate([on["n"], equal["n"]])
    f_all = np.concatenate([on["mean"], equal["frac"]])
    ax.plot(n_all, f_all, "o-", color=C_EQUAL, ms=4.0, lw=1.25, zorder=3,
            label="equal powers")
    ax.plot(spread["n"], spread["frac"], "s--", color=C_SPREAD, ms=4.0, lw=1.2,
            label=r"decay $1.0\to0.3$")
    ax.plot(weak["n"], weak["frac"], "^:", color=C_WEAK, ms=4.4, lw=1.2,
            label=r"one weak, $\kappa=0.01$")

    ax.axvline(n_th, color=C_BOUND, ls=(0, (1.5, 1.5)), lw=1.0, zorder=2)
    ax.annotate(r"sufficient bound", xy=(n_th, 0.55), xytext=(n_th * 1.7, 0.55),
                fontsize=6.0, color=C_BOUND, va="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=C_BOUND))

    ax.axvline(round_symbols, color=C_WEAK, ls="-.", lw=0.85, zorder=2)
    ax.annotate("one round", xy=(round_symbols, 0.34),
                xytext=(round_symbols * 0.5, 0.34), fontsize=6.0,
                color=C_WEAK, va="center", ha="right",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=C_WEAK))

    ax.set_xlim(80, 1.5e9)
    ax.set_ylim(-0.09, 1.95)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel(r"observed symbols  $n$", fontsize=8)
    ax.set_ylabel("codes recovered", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.0, loc="upper left", frameon=False, handlelength=1.6,
              borderaxespad=0.2, labelspacing=0.28, borderpad=0.1)
    ax.set_title(r"(a)  from $\hat\Sigma$ alone,  $L=1024$, $M=8$", fontsize=8)

    # ---------------- (b) cost of refreshing -----------------------------
    curve = e12["curve"]
    T = np.asarray(curve["T"], dtype=float)
    cost = np.asarray(curve["prf_bits_per_symbol"], dtype=float)
    bx.set_xscale("log")
    bx.set_yscale("log")
    bx.plot(T, cost, "-", color="#6c3483", lw=1.35)
    brows = e12["measured"]
    t_on = [float(r["T"]) for r in brows if float(r["recovery_fraction"]) > 0]
    if t_on:
        bx.axvspan(min(t_on), max(float(r["T"]) for r in brows),
                   color=C_WEAK, alpha=0.12, lw=0)
        bx.text(np.sqrt(min(t_on) * max(float(r["T"]) for r in brows)), 3e-4,
                "attack\nsucceeds", fontsize=5.8, color=C_WEAK, ha="center",
                va="bottom")
    bx.set_xlabel(r"refresh period  $T$  (symbols/codebook)", fontsize=8)
    bx.set_ylabel(r"PRF rate  $ML/T$  (bit/sym)", fontsize=8)
    bx.tick_params(labelsize=7)
    bx.grid(alpha=0.20, which="major", lw=0.5)
    bx.set_title("(b)  cost of the only defence", fontsize=8)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=300)
    plt.close(fig)

    t_first = on["n"][np.argmax(on["mean"] > 0)]
    t_full = on["n"][np.argmax(on["mean"] >= 1.0)]
    print(f"wrote {OUT}")
    print(f"onset: {t_first:.0f} -> {t_full:.0f} symbols; "
          f"n_th = {n_th:.4g}; looseness = {n_th / t_full:.0f}x")
    print(f"fraction of a round at onset: "
          f"{t_first / round_symbols:.2e}")


if __name__ == "__main__":
    main()
