"""E6: Composable security parameters -- the epsilon budget.

Panel 1: bar chart of every component of eps_total, in -log2 units.
Panel 2: eps_total vs number of rounds for union-bound and sequential
         composition, against the target eps=1e-9.
Panel 3: leftover-hash budget: how much of K_n survives the extractor for a
         range of H_min(K_n)/|K_n| ratios.

Outputs
-------
results/figures/e6_epsilon.png
results/data/e6_epsilon.json
"""

from __future__ import annotations

import math

import numpy as np

from _common import dump, header, scenario

from qcdma_qkd import plotting, security


def main() -> dict:
    header("E6  Composable security-parameter budget")
    sc = scenario()
    sec = sc["sec"]
    b = security.from_config(sec)
    table = b.table()
    components = {k: v for k, v in table.items() if k != "total"}

    for k, v in components.items():
        print(f"  {k:9s} {v:.2e}   (-log2 = {-math.log2(v):5.1f})")
    print(f"  {'TOTAL':9s} {table['total']:.2e}   "
          f"(-log2 = {-math.log2(table['total']):5.1f})   "
          f"target {sec.eps_total:.0e} -> "
          f"{'PASS' if table['total'] <= sec.eps_total else 'FAIL'}")

    rounds = np.array([1, 10, 100, 1000, 10 ** 4, 10 ** 5])
    eps_round = 1e-12
    seq = np.array([security.compose_sequential(eps_round, int(n)) for n in rounds])
    uni = np.array([security.compose_sum([eps_round] * int(n)) for n in rounds])

    ratios = np.linspace(0.5, 0.99, 50)
    k_bits = 1e7
    lhl = np.array([security.lhl_output_bits(r * k_bits, 1e-12) for r in ratios])

    print(f"  sequential composition over 1e4 rounds: {seq[4]:.2e}  "
          f"(union bound would give {uni[4]:.2e})")

    # --- plot -------------------------------------------------------------- #
    fig, axes = plotting.plt.subplots(1, 3, figsize=(12.6, 3.4))

    ax = axes[0]
    names = list(components)
    vals = [-math.log2(max(v, 1e-300)) for v in components.values()]
    ax.barh(names, vals, color="#2471a3", alpha=0.85)
    ax.axvline(-math.log2(sec.eps_total), color="#c0392b", ls="--",
               label=f"target {-math.log2(sec.eps_total):.1f}")
    ax.set_xlabel(r"$-\log_2 \epsilon_i$")
    ax.set_title("Component security parameters")
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    ax = axes[1]
    ax.loglog(rounds, seq, "o-", label="sequential (diamond norm)")
    ax.loglog(rounds, uni, "s--", label="union bound")
    ax.axhline(sec.eps_total, color="#c0392b", ls=":", label=r"target $10^{-9}$")
    ax.set_xlabel("number of rounds")
    ax.set_ylabel(r"$\epsilon_{\mathrm{total}}$")
    ax.set_title("Composition over rounds")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_ylim(1e-12, 5e-7)

    ax = axes[2]
    ax.plot(ratios, lhl / 1e6, color="#1e8449", label="extractable")
    ax.plot(ratios, ratios * k_bits / 1e6, "k:", lw=1, label=r"$H_{\min}$")
    ax.set_xlabel(r"$H_{\min}(K_n)\,/\,|K_n|$")
    ax.set_ylabel("extractable key (Mbit)")
    ax.set_title(r"Leftover-hash budget ($\epsilon_{\mathrm{kdf}}=10^{-12}$)")
    ax.legend(fontsize=7, loc="upper left")
    plotting.annotate(ax, "cost: 2 log2(1/eps) = 80 bit\n(~0.1 Mbit here)",
                      xy=(0.38, 0.20), fontsize=7)

    plotting.save(fig, "e6_epsilon")

    return dict(components=components, total=table["total"],
                target=sec.eps_total, rounds=rounds,
                eps_sequential=seq, eps_union=uni,
                ratios=ratios, lhl_bits=lhl, k_bits=k_bits)


if __name__ == "__main__":
    dump("e6_epsilon", main())
