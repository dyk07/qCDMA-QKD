"""E5: Key budget -- raw vs net application rate, and the net-rate phase diagram.

Panel 1: raw key rate, auth consumption, seed consumption, net rate vs distance
         (with and without the MUI excess noise charged in).
Panel 2: net-rate phase diagram vs (distance, authentication message count).
Panel 3: share of the round key spent on auth+seed vs round length.

Outputs
-------
results/figures/e5_budget.png
results/data/e5_budget.json
"""

from __future__ import annotations

import numpy as np

from _common import dump, header, rate_at, rate_fn, scenario

from qcdma_qkd import budget, cvqkd, mui, plotting


def main() -> dict:
    header("E5  Key budget of the closed loop")
    sc = scenario()
    cfg, bcfg, cdma = sc["cv"], sc["budget"], sc["cdma"]
    n_symbols = 10 ** 8
    symbol_rate = cdma.symbol_rate()   # per-user: chip_rate / L

    # MUI charge (worst case, 8 users, L chips)
    kappa = mui.uniform_kappa(cdma.n_users - 1, 1.0)
    xi_mui = mui.required_xi(1e-12, kappa, cdma.spreading_factor)

    dist = np.arange(5.0, 125.0, 2.5)
    fin_kw = dict(eps_sm=sc["sec"].eps_sm, eps_pa=sc["sec"].eps_pa,
                  sample_fraction=bcfg.pe_sample_fraction)
    # finite-size rates: the loop uses these, so the budget picture must too
    clean = budget.budget_vs_distance(dist, rate_fn(cfg, symbol_rate, 0.0,
                                                    n_symbols, **fin_kw),
                                      bcfg, n_symbols, symbol_rate)
    withmui = budget.budget_vs_distance(dist, rate_fn(cfg, symbol_rate, xi_mui,
                                                      n_symbols, **fin_kw),
                                        bcfg, n_symbols, symbol_rate)

    # Phase diagram: how many authenticated messages per round can we afford?
    #   u >= 0  <=>  a(n_msg) + seed <= K
    # A key serves up to ``reuse`` messages and keys are not splittable, so the
    # marginal cost of one more *round* is a whole ``key_length``, NOT the
    # amortised ``per_message`` (which is l/reuse and would overstate the
    # affordable count by a factor reuse ~ 1572).
    per_round_auth = budget.auth_key_bits(bcfg)["key_length"]
    msg_grid = np.array([8, 32, 128, 512, 2048, 8192])
    affordable = []
    for L in dist:
        k_bits = rate_at(cfg, float(L), xi_mui, n_symbols, **fin_kw)["S_pos"] * n_symbols
        affordable.append(int((k_bits - bcfg.seed_bits)
                              // per_round_auth) * budget.auth_key_bits(bcfg)["reuse"])
    affordable = np.asarray(affordable, dtype=float)
    # feasibility matrix: message demand (rows) vs supply (distance)
    phase = (msg_grid[:, None] <= affordable[None, :]).astype(float)

    # overhead vs round length
    lens = np.array([1e6, 1e7, 1e8, 1e9])
    ovh = []
    for n in lens:
        r = rate_at(cfg, 25.0, xi_mui, n, **fin_kw)
        k = r["S_pos"] * n
        ovh.append(budget.overhead_ratio(int(n), r["I_AB"], k, bcfg))

    print(f"MUI charge: xi_MUI(eps=1e-12, 8 users, L={cdma.spreading_factor}) = "
          f"{xi_mui:.4f} SNU")
    print(f"  max reach, clean      : {cvqkd.max_distance(cfg):.1f} km")
    print(f"  max reach, with MUI   : {cvqkd.max_distance(cfg, xi_mui):.1f} km")
    print(f"  overhead at 25 km, 1e8 symbols: {ovh[2]:.3e} of K")

    # --- plot -------------------------------------------------------------- #
    fig, axes = plotting.plt.subplots(1, 3, figsize=(12.6, 3.4))

    ax = axes[0]
    ax.plot(dist, clean["raw"] / 1e6, color="#7f8c8d", lw=1, ls=":", label="raw $K_n$ (clean)")
    ax.plot(dist, clean["net"] / 1e6, color="#2471a3", label="net $u_n$ (clean)")
    ax.plot(dist, withmui["raw"] / 1e6, color="#7f8c8d", lw=1, ls=":", label="raw $K_n$ (+MUI)")
    ax.plot(dist, withmui["net"] / 1e6, color="#c0392b", label="net $u_n$ (+MUI)")
    ax.set_xlabel("fibre length (km)")
    ax.set_ylabel("rate (Mbit/s)")
    ax.set_title("Net application rate")
    ax.legend(fontsize=6.5)

    ax = axes[1]
    ax.imshow(phase, origin="lower", aspect="auto", cmap="RdYlGn",
              extent=[dist[0], dist[-1], -0.5, len(msg_grid) - 0.5],
              vmin=0, vmax=1)
    ax.set_yticks(range(len(msg_grid)))
    ax.set_yticklabels(msg_grid)
    ax.set_xlabel("fibre length (km)")
    ax.set_ylabel("authenticated messages / round")
    ax.set_title("Feasible region (green = affordable)")
    ax2 = ax.twinx()
    ax2.semilogy(dist, affordable, color="k", lw=1.4, ls="--",
                 label="max affordable messages")
    ax2.set_ylabel("max affordable messages", fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.grid(False)
    ax2.legend(fontsize=6.5, loc="lower left")

    ax = axes[2]
    ax.loglog(lens, ovh, "o-", color="#1e8449")
    ax.set_xlabel("round length (symbols)")
    ax.set_ylabel("(auth + seed) / $K_n$")
    ax.set_title("Overhead shrinks with block length")
    plotting.annotate(ax, "auth cost is per-message,\nnot per-bit -> long rounds win",
                      xy=(0.05, 0.9), fontsize=7)

    plotting.save(fig, "e5_budget")

    return dict(dist=dist, clean=clean, with_mui=withmui, xi_mui=xi_mui,
                msg_grid=msg_grid, phase=phase, affordable=affordable,
                per_message_bits=budget.auth_key_bits(bcfg)["per_message"],
                per_round_auth_bits=per_round_auth,
                lens=lens, overhead=ovh)


if __name__ == "__main__":
    dump("e5_budget", main())
