"""E1: CV-QKD baseline -- asymptotic vs finite-size rate, and the effect of
excess noise on maximum reach.

Outputs
-------
results/figures/e1_cvqkd_baseline.png
results/data/e1_cvqkd_baseline.json
"""

from __future__ import annotations

import numpy as np

from _common import dump, header, rate_at, scenario

from qcdma_qkd import cvqkd, plotting


def main() -> dict:
    header("E1  CV-QKD baseline: asymptotic vs finite-size")
    sc = scenario()
    cfg = sc["cv"]
    distances = np.arange(0.0, 130.0, 1.0)

    asym, finite = [], []
    for L in distances:
        asym.append(rate_at(cfg, float(L))["S_pos"])
        T = cvqkd.total_transmittance(float(L), cfg.alpha_db_per_km, cfg.eta)
        r = cvqkd.finite_size_rate(cfg.V_A, T, cfg.excess_noise, cfg.eta,
                                   cfg.v_el, cfg.beta, cfg.detection,
                                   n_symbols=10 ** 8, eps_sm=1e-10,
                                   eps_pa=1e-12, sample_fraction=0.1)
        finite.append(r["S_pos"])
    asym, finite = np.asarray(asym), np.asarray(finite)

    # --- excess noise sensitivity ---------------------------------------- #
    xi_grid = np.array([0.005, 0.01, 0.02, 0.05, 0.10, 0.20])
    reach = np.array([cvqkd.max_distance(cfg, extra_noise=float(x)) for x in xi_grid])

    print(f"padding  asym rate @0km = {asym[0]:.4f} bits/symbol")
    print(f"         finite rate @0km = {finite[0]:.4f} bits/symbol")
    print(f"         asym max distance = {cvqkd.max_distance(cfg):.1f} km")
    print(f"         if MUI adds xi=0.055 -> max distance "
          f"{cvqkd.max_distance(cfg, 0.0546):.1f} km")

    # --- plot ------------------------------------------------------------- #
    fig, axes = plotting.plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    # bits/symbol * symbol_rate -> bits/s; show Mbit/s
    ax.plot(distances, asym * cfg.symbol_rate / 1e6, label="asymptotic")
    ax.plot(distances, finite * cfg.symbol_rate / 1e6, "--",
            label="finite size (n=1e8)")
    ax.set_xlabel("fibre length (km)")
    ax.set_ylabel("secret key rate (Mbit/s)")
    ax.set_title("GG02 homodyne, reverse reconciliation")
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.semilogy(xi_grid, reach, "o-")
    ax.set_xlabel(r"channel excess noise $\xi$ (SNU)")
    ax.set_ylabel("maximum reach (km)")
    ax.set_title("Reach collapses with excess noise")
    ax.set_xscale("log")

    plotting.save(fig, "e1_cvqkd_baseline")

    return dict(distances=distances, asym=asym, finite=finite,
                xi_grid=xi_grid, reach=reach,
                asym_max_distance=cvqkd.max_distance(cfg),
                reach_at_mui=cvqkd.max_distance(cfg, 0.0546))


if __name__ == "__main__":
    dump("e1_cvqkd_baseline", main())
