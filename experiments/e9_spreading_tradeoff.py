"""E9: Spreading-factor trade-off -- L cuts the rate twice.

Once the rate bookkeeping is correct (``symbol_rate = chip_rate / L``), the
spreading factor appears on *both* sides of the ledger:

* increasing ``L`` **suppresses MUI**   (``xi_MUI ~ 1/L``, pushes the rate up)
* increasing ``L`` **divides the symbol rate** (``1/L``, pushes the rate down)

So for a fixed hardware chip rate there is an **optimal** ``L``, and it depends
on distance and on the number of users -- both because they move the operating
point along the steep part of the CV-QKD rate curve.

Panels
------
1. Bit rate vs L for several distances: the peak, and how flat it is.
2. Bit rate vs L for several user counts at 25 km.
3. Optimal L and its rate vs distance, with the MUI-limited and rate-limited
   asymptotes marked.

Outputs
-------
results/figures/e9_spreading_tradeoff.png
results/data/e9_spreading_tradeoff.json
"""

from __future__ import annotations

import numpy as np

from _common import dump, header, rate_at, scenario

from qcdma_qkd import mui, plotting

L_GRID = np.array([128, 256, 512, 1024, 2048, 4096, 8192, 16384])


def bit_rate_at(cfg, r_chip: float, L: int, n_users: int, distance_km: float,
                eps_mui: float = 1e-12) -> float:
    """Per-user bit rate (bit/s) for a given spreading factor."""
    kappa = mui.uniform_kappa(n_users - 1, 1.0)
    xi = mui.required_xi(eps_mui, kappa, int(L))
    s = rate_at(cfg, distance_km, xi)["S_pos"]
    return s * (r_chip / float(L))


def main() -> dict:
    header("E9  Spreading-factor trade-off (L divides the rate)")
    sc = scenario()
    cfg, cdma = sc["cv"], sc["cdma"]
    rc = cdma.chip_rate
    users = cdma.n_users

    # --- 1. vs distance ---------------------------------------------------- #
    distances = (5.0, 10.0, 25.0, 50.0)
    by_dist = {}
    for d in distances:
        by_dist[d] = np.array([bit_rate_at(cfg, rc, int(L), users, d)
                               for L in L_GRID])

    # --- 2. vs user count --------------------------------------------------- #
    user_grid = (2, 4, 8, 16)
    by_users = {}
    for m in user_grid:
        by_users[m] = np.array([bit_rate_at(cfg, rc, int(L), m, 25.0)
                                for L in L_GRID])

    # --- 3. optimal L vs distance ------------------------------------------ #
    dist_fine = np.arange(5.0, 65.0, 2.5)
    l_opt, r_opt = [], []
    for d in dist_fine:
        rates = np.array([bit_rate_at(cfg, rc, int(L), users, float(d))
                          for L in L_GRID])
        i = int(np.argmax(rates))
        l_opt.append(float(L_GRID[i]))
        r_opt.append(float(rates[i]))

    print(f"chip rate {rc / 1e9:.0f} Gchip/s, {users} users, "
          f"eps_MUI = 1e-12")
    i1024 = int(np.where(L_GRID == 1024)[0][0])
    i4096 = int(np.where(L_GRID == 4096)[0][0])
    for d in distances:
        i = int(np.argmax(by_dist[d]))
        print(f"  {d:4.0f} km: L* = {L_GRID[i]:5d}  "
              f"rate = {by_dist[d][i] / 1e3:8.2f} kbit/s   "
              f"(L=1024: {by_dist[d][i1024] / 1e3:7.2f}, "
              f"L=4096: {by_dist[d][i4096] / 1e3:7.2f})")
    for m in user_grid:
        i = int(np.argmax(by_users[m]))
        print(f"  {m:2d} users @25km: L* = {L_GRID[i]:5d}  "
              f"rate = {by_users[m][i] / 1e3:8.2f} kbit/s")

    # --- plot -------------------------------------------------------------- #
    fig, axes = plotting.plt.subplots(1, 3, figsize=(13.4, 3.5))
    fig.subplots_adjust(wspace=0.34)

    ax = axes[0]
    for d, c in zip(distances, ("#1e8449", "#2471a3", "#d68910", "#c0392b")):
        ax.semilogx(L_GRID, by_dist[d] / 1e3, "o-", color=c, label=f"{d:.0f} km")
    ax.set_xlabel("spreading factor L (chips/symbol)")
    ax.set_ylabel("bit rate per user (kbit/s)")
    ax.set_title("vs distance (8 users)")
    ax.legend(fontsize=7)

    ax = axes[1]
    for m, c in zip(user_grid, ("#1e8449", "#2471a3", "#d68910", "#c0392b")):
        ax.semilogx(L_GRID, by_users[m] / 1e3, "o-", color=c, label=f"{m} users")
    ax.set_xlabel("spreading factor L (chips/symbol)")
    ax.set_ylabel("bit rate per user (kbit/s)")
    ax.set_title("vs user count (25 km)")
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.plot(dist_fine, l_opt, "o-", color="#2471a3", ms=3, label=r"$L^\ast$")
    ax.set_xlabel("fibre length (km)")
    ax.set_ylabel(r"optimal $L^\ast$", color="#2471a3")
    ax.set_ylim(0, 2.4e4)
    ax2 = ax.twinx()
    ax2.plot(dist_fine, np.asarray(r_opt) / 1e3, "s--", color="#c0392b", ms=3)
    ax2.set_ylabel("rate at $L^\\ast$ (kbit/s)", color="#c0392b")
    ax2.grid(False)
    ax.set_title(r"$L^\ast$ grows once MUI dominates")
    h1, l1 = ax.get_legend_handles_labels()
    ax.legend(h1, l1, fontsize=7, loc="upper right")

    plotting.save(fig, "e9_spreading_tradeoff")

    return dict(L_grid=L_GRID, chip_rate=rc, n_users=users,
                by_distance={str(d): by_dist[d] for d in distances},
                by_users={str(m): by_users[m] for m in user_grid},
                dist_fine=dist_fine, L_opt=l_opt, rate_opt=r_opt)


if __name__ == "__main__":
    dump("e9_spreading_tradeoff", main())
