"""E2: Multi-user interference -- why the *block-averaged* xi matters.

Three panels:

1. xi_MUI (mean and eps-quantile) vs number of users, for two spreading factors.
2. Histogram of block-averaged xi vs the best Gaussian fit: the distribution is
   Gamma-like, and its relative standard deviation is sqrt(2/K) -- it does not
   shrink with block length.
3. Key rate at 20 km as a function of L, provisioning for the eps=1e-12
   quantile vs the mean.

Outputs
-------
results/figures/e2_mui.png
results/data/e2_mui.json
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from _common import dump, header, rate_at, scenario

from qcdma_qkd import mui, plotting


def main() -> dict:
    header("E2  Dynamic MUI: block-averaged excess noise")
    sc = scenario()
    cfg = sc["cv"]
    V_A = cfg.V_A

    # --- 1. xi vs number of users ----------------------------------------- #
    users = np.arange(2, 33)
    curves = {}
    for L in (256, 1024, 4096):
        curves[L] = mui.mui_vs_users(users, L, V_A, 1.0, eps_mui=1e-12)

    # --- 2. distribution of the block-averaged xi ------------------------- #
    L0, K = 1024, 7
    kappa = mui.uniform_kappa(K, 1.0)
    xi_mean = mui.xi_block_mean(kappa, L0)
    xi_req = mui.required_xi(1e-12, kappa, L0)
    samples = mui.xi_block_samples(kappa, L0, n_draws=200_000, seed=7)
    rel_std = mui.xi_block_std(kappa, L0) / xi_mean

    # --- 3. rate versus L -------------------------------------------------- #
    L_grid = np.array([256, 512, 1024, 2048, 4096, 8192])
    r_mean, r_req = [], []
    for L in L_grid:
        k = mui.uniform_kappa(K, 1.0)
        r_mean.append(rate_at(cfg, 20.0, mui.xi_block_mean(k, L))["S_pos"])
        r_req.append(rate_at(cfg, 20.0, mui.required_xi(1e-12, k, L))["S_pos"])
    r_mean, r_req = np.asarray(r_mean), np.asarray(r_req)

    print(f"K={K}, L={L0}: xi_mean={xi_mean:.4f}  xi(1e-12)={xi_req:.4f}  "
          f"ratio={xi_req / xi_mean:.2f}x")
    print(f"  relative std of xi_block = {rel_std:.3f} (theory sqrt(2/K)="
          f"{np.sqrt(2 / K):.3f})")
    print(f"  rate @20km: mean-provisioned {r_mean[2]:.4f}, "
          f"tail-provisioned {r_req[2]:.4f} bits/symbol")

    # --- plot -------------------------------------------------------------- #
    fig, axes = plotting.plt.subplots(1, 3, figsize=(12.6, 3.4))

    ax = axes[0]
    for L, c in zip((256, 1024, 4096), ("#c0392b", "#2471a3", "#1e8449")):
        ax.plot(users, curves[L]["xi"], color=c, label=f"L={L} (mean)")
        ax.plot(users, curves[L]["xi_required"], color=c, ls="--",
                label=f"L={L} (eps=1e-12)")
    ax.set_xlabel("number of users M")
    ax.set_ylabel(r"$\xi_{\mathrm{MUI}}$ (SNU)")
    ax.set_title("MUI excess noise vs users")
    ax.legend(fontsize=6.5, ncol=2)

    ax = axes[1]
    ax.hist(samples, bins=120, density=True, color="#2471a3", alpha=0.55,
            label="block-averaged $\\xi$ (MC)")
    xs = np.linspace(samples.min(), samples.max(), 400)
    ax.plot(xs, stats.norm.pdf(xs, samples.mean(), samples.std()), "k--",
            label="Gaussian fit")
    ax.axvline(xi_mean, color="k", lw=1)
    ax.axvline(xi_req, color="#c0392b", lw=1.4, ls="--")
    plotting.annotate(ax, f"rel. std = {rel_std:.2f}\n(does not vanish with n)",
                      xy=(0.55, 0.95), fontsize=7)
    ax.set_xlabel(r"block-averaged $\xi$ (SNU)")
    ax.set_ylabel("density")
    ax.set_title(f"K={K}, L={L0}: variance is *not* averaged away")
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.plot(L_grid, r_mean, "o-", label="provision for mean")
    ax.plot(L_grid, r_req, "s--", label="provision for eps=1e-12 quantile")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("spreading factor L (chips/symbol)")
    ax.set_ylabel("key rate at 20 km (bits/symbol)")
    ax.set_title("Cost of the security margin")
    ax.legend(fontsize=7)

    plotting.save(fig, "e2_mui")

    return dict(users=users,
                xi_by_L={str(L): curves[L] for L in curves},
                L0=L0, K=K, xi_mean=xi_mean, xi_required=xi_req,
                rel_std=rel_std, samples_n=len(samples),
                L_grid=L_grid, rate_mean=r_mean, rate_required=r_req)


if __name__ == "__main__":
    dump("e2_mui", main())
