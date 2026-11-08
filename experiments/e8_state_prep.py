"""E8: State-preparation leakage -- the layer privacy amplification cannot fix.

The paper's Section 5 distinguishes two leakage channels:

* the *classical* layer: PA truncates Eve's accumulated knowledge of K_n each
  round, so the loop is "reset" round by round;
* the *quantum-preparation* layer: if the spreading code modulates the phase of
  the transmitted state, partial code knowledge shrinks Eve's uncertainty about
  the *prepared* ensemble.  PA acts on the key, not on the state, so this term
  must be provisioned in the rate.

Panels
------
1. xi_prep = g^2 V_A / (2L) vs the leaked fraction g, for several L.
2. Rate loss at 25 km vs g, for the same L values: shows how much code secrecy
   is really needed.
3. Tolerance region: the largest g that keeps the rate loss under 1%, 5%, 10%.

Outputs
-------
results/figures/e8_state_prep.png
results/data/e8_state_prep.json
"""

from __future__ import annotations

import numpy as np

from _common import dump, header, rate_at, scenario

from qcdma_qkd import attack, mui, plotting


def main() -> dict:
    header("E8  State-preparation leakage (the layer PA cannot absorb)")
    sc = scenario()
    cfg, cdma = sc["cv"], sc["cdma"]
    L_list = (256, 1024, 4096)
    g_grid = np.linspace(0.0, 1.0, 51)

    xi_mui = mui.required_xi(1e-12, mui.uniform_kappa(cdma.n_users - 1, 1.0),
                             cdma.spreading_factor)
    base_rate = rate_at(cfg, 25.0, xi_mui)["S_pos"]

    xi_prep = {L: np.array([attack.state_preparation_leak(g, cfg.V_A, L)["xi_prep"]
                            for g in g_grid]) for L in L_list}
    rates = {L: np.array([rate_at(cfg, 25.0, xi_mui + x)["S_pos"]
                          for x in xi_prep[L]]) for L in L_list}

    # tolerance: largest g with relative rate loss below a threshold
    tols = {}
    for L in L_list:
        row = {}
        for loss in (0.01, 0.05, 0.10):
            ok = np.where((base_rate - rates[L]) / base_rate <= loss)[0]
            row[f"{loss:.2f}"] = float(g_grid[ok.max()]) if ok.size else 0.0
        tols[L] = row

    print(f"baseline rate at 25 km (with MUI xi={xi_mui:.4f}): {base_rate:.4f} bits/symbol")
    for L in L_list:
        print(f"  L={L:5d}: xi_prep(g=1)={xi_prep[L][-1]:.5f}, "
              f"rate(g=1)={rates[L][-1]:.4f} "
              f"({100 * (1 - rates[L][-1] / base_rate):.2f}% loss); "
              f"g tolerance for 1%/5%/10% loss = "
              f"{tols[L]['0.01']:.2f}/{tols[L]['0.05']:.2f}/{tols[L]['0.10']:.2f}")

    # --- plot -------------------------------------------------------------- #
    fig, axes = plotting.plt.subplots(1, 3, figsize=(12.6, 3.4))

    ax = axes[0]
    for L, col in zip(L_list, ("#c0392b", "#2471a3", "#1e8449")):
        ax.plot(g_grid, xi_prep[L], color=col, label=f"L={L}")
    ax.set_xlabel("leaked code fraction g")
    ax.set_ylabel(r"$\xi_{\mathrm{prep}}$ (SNU)")
    ax.set_title(r"Preparation noise $\propto g^2 V_A / 2L$")
    ax.legend(fontsize=7)

    ax = axes[1]
    for L, col in zip(L_list, ("#c0392b", "#2471a3", "#1e8449")):
        ax.plot(g_grid, 100 * (rates[L] / base_rate - 1), color=col, label=f"L={L}")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(-5, color="#7f8c8d", ls=":", lw=1, label="5% budget")
    ax.set_xlabel("leaked code fraction g")
    ax.set_ylabel("rate change (%)")
    ax.set_title("Key rate cost at 25 km")
    ax.legend(fontsize=7)

    ax = axes[2]
    x = np.arange(len(L_list))
    w = 0.26
    for i, (loss, col) in enumerate(zip(("0.01", "0.05", "0.10"),
                                        ("#c0392b", "#d68910", "#1e8449"))):
        vals = [tols[L][loss] for L in L_list]
        ax.bar(x + (i - 1) * w, vals, w, color=col, alpha=0.85,
               label=f"{float(loss) * 100:.0f}% rate budget")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L={L}" for L in L_list])
    ax.set_ylabel("max tolerable g")
    ax.set_title("How secret must the code be?")
    ax.legend(fontsize=7)

    plotting.save(fig, "e8_state_prep")

    return dict(L_list=list(L_list), g_grid=g_grid,
                xi_prep={str(k): v for k, v in xi_prep.items()},
                rates={str(k): v for k, v in rates.items()},
                tolerance={str(k): v for k, v in tols.items()},
                base_rate=base_rate, xi_mui=xi_mui, V_A=cfg.V_A)


if __name__ == "__main__":
    dump("e8_state_prep", main())
