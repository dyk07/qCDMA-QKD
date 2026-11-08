"""E4: Cross-round leakage -- the refresh / seed / spreading-factor constraint.

Panel 1: min-entropy of the *current* code seed after one round of leakage, as a
         function of the refresh fraction, for three seed lengths (extractor
         model).  The x-intercepts are the design constraint f > 1 - seed/(L p).
Panel 2: eps_code vs refresh fraction on a log scale, both extractor and
         computational-PRF models.
Panel 3: the allowed chips-per-seed-bit ratio: a feasible region for (L, seed).

Outputs
-------
results/figures/e4_refresh.png
results/data/e4_refresh.json
"""

from __future__ import annotations

import numpy as np

from _common import dump, header, scenario

from qcdma_qkd import attack, plotting


def main() -> dict:
    header("E4  Cross-round leakage and the code-refresh constraint")
    sc = scenario()
    L = sc["cdma"].spreading_factor

    f_grid = np.linspace(0.0, 1.0, 101)
    seeds = (128, 256, 512)

    # --- 1/2. seed min-entropy and eps_code ------------------------------- #
    h = {s: np.array([attack.residual_seed_entropy(s, L, (1 - f) * 1.0) for f in f_grid])
         for s in seeds}
    eps = {s: np.array([attack.cross_round_eps(f, s, L, 1.0) for f in f_grid])
           for s in seeds}
    eps_prf = 2.0 ** (-256)

    constraint = {s: attack.refresh_security_constraint(s, L) for s in seeds}
    for s, c in constraint.items():
        print(f"seed={s:4d} bits, L={L}: f_min = {c['f_min']:.3f} "
              f"(known chips must stay < {s})")

    # --- 3. feasible region ----------------------------------------------- #
    L_grid = np.array([128, 256, 512, 1024, 2048, 4096])
    seed_grid = np.array([128, 256, 512, 1024])
    feasible = np.zeros((len(seed_grid), len(L_grid)))
    for i, s in enumerate(seed_grid):
        for j, LL in enumerate(L_grid):
            feasible[i, j] = attack.refresh_security_constraint(s, int(LL))["f_min"]

    # --- where do we sit by default? -------------------------------------- #
    d = constraint[256]
    print(f"default: seed=256, L={L} -> f_min={d['f_min']:.2f}; Eve may know at "
          f"most {d['max_known_chips']:.0f} chips of the {L} chips in a round")

    # --- plot -------------------------------------------------------------- #
    fig, axes = plotting.plt.subplots(1, 3, figsize=(13.6, 3.6))
    fig.subplots_adjust(wspace=0.42)

    ax = axes[0]
    # shade only the strictest constraint so the plot does not turn muddy
    ax.axvspan(0.0, max(constraint[s]["f_min"] for s in seeds),
               color="#c0392b", alpha=0.08,
               label="seed fully exposed")
    for s, c in zip(seeds, ("#c0392b", "#2471a3", "#1e8449")):
        ax.plot(f_grid, h[s], color=c, label=f"seed {s} bit")
        ax.axvline(constraint[s]["f_min"], color=c, ls=":", lw=1)
    ax.set_xlabel("refresh fraction f")
    ax.set_ylabel(r"$H_{\min}(c_n)$ (bits)")
    ax.set_title("Seed entropy after one leaked round")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_ylim(-20, 610)

    ax = axes[1]
    for s, c in zip(seeds, ("#c0392b", "#2471a3", "#1e8449")):
        ax.semilogy(f_grid, np.clip(eps[s], 1e-300, 1), color=c,
                    label=f"extractor, {s} bit")
    ax.axhline(eps_prf, color="k", ls="--", lw=1.2, label="computational PRF")
    ax.set_xlabel("refresh fraction f")
    ax.set_ylabel(r"$\epsilon_{\mathrm{code}}$", labelpad=1)
    ax.set_title("Extractor vs PRF: the price of\ninformation-theoretic codes")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_ylim(1e-150, 3)

    ax = axes[2]
    ax = axes[2]
    feasible_masked = np.ma.masked_where(feasible >= 1.0, feasible)
    cmap = plotting.plt.get_cmap("viridis").copy()
    cmap.set_bad("#d5d8dc")          # infeasible cells: neutral grey
    im = ax.imshow(feasible_masked, origin="lower", aspect="auto", cmap=cmap,
                   vmin=0, vmax=1)
    ax.set_xticks(range(len(L_grid)))
    ax.set_xticklabels(L_grid)
    ax.set_yticks(range(len(seed_grid)))
    ax.set_yticklabels(seed_grid)
    ax.set_xlabel("spreading factor L")
    ax.set_ylabel("seed length (bits)")
    ax.set_title(r"required refresh $f_{\min}$" + "\n(grey = infeasible)")
    for i in range(len(seed_grid)):
        for j in range(len(L_grid)):
            v = feasible[i, j]
            ax.text(j, i, "N/A" if v >= 1.0 else f"{v:.2f}",
                    ha="center", va="center", fontsize=6.5,
                    color="k" if v >= 1.0 else ("w" if v < 0.55 else "k"))
    fig.colorbar(im, ax=ax, fraction=0.046)

    plotting.save(fig, "e4_refresh")

    return dict(f_grid=f_grid, L=L, seeds=list(seeds),
                h_min={str(s): h[s] for s in seeds},
                eps_code={str(s): np.clip(eps[s], 1e-300, 1) for s in seeds},
                eps_prf=eps_prf,
                constraint={str(s): c for s, c in constraint.items()},
                L_grid=L_grid, seed_grid=seed_grid, f_min_map=feasible)


if __name__ == "__main__":
    dump("e4_refresh", main())
