"""E7: End-to-end closed-loop simulation.

Panel 1: per-round key budget breakdown (K_n, a_n, c_n, u_n) over 20 rounds,
         with an inset showing the overhead a_n+c_n on a log scale.
Panel 2: what the loop does under a MitM / tamper at round 8 -- it aborts before
         any quantum resources are spent, versus the unauthenticated case which
         runs forever producing useless keys.
Panel 3: the bootstrap cost K_0 as a function of distance and control-message
         count: it is negligible compared with the round key, which is the
         quantitative answer to "how much trust does round 0 really need?"

Outputs
-------
results/figures/e7_loop.png
results/data/e7_loop.json
"""

from __future__ import annotations

import numpy as np

from _common import dump, header, rate_at, scenario

from qcdma_qkd import attack, budget, loop, mui, plotting


def main() -> dict:
    header("E7  Closed-loop protocol simulation")
    sc = scenario()
    cfg, cdma, bcfg = sc["cv"], sc["cdma"], sc["budget"]
    n_rounds = 20
    n_symbols = 10 ** 8

    res = loop.run_loop(n_rounds, k0_bits=2 ** 20, cfg_cv=cfg, cfg_cdma=cdma,
                        cfg_bud=bcfg, distance_km=25.0, n_symbols=n_symbols)
    summ = loop.loop_summary(res)
    tampered = loop.run_loop(n_rounds, k0_bits=2 ** 20, cfg_cv=cfg,
                             cfg_cdma=cdma, cfg_bud=bcfg, distance_km=25.0,
                             n_symbols=n_symbols, tamper_at=8)

    rounds = np.array([r.round_index for r in res["rounds"]])
    k = np.array([r.k_bits for r in res["rounds"]])
    a = np.array([r.a_bits for r in res["rounds"]])
    c = np.array([r.c_bits for r in res["rounds"]])
    u = np.array([r.u_bits for r in res["rounds"]])
    xi_mui = res["rounds"][0].xi_mui

    # K_0 accounting: only round 1 draws on it
    k0 = budget.k0_budget(bcfg, n_symbols, res["rounds"][0].extras["I_AB"],
                          bootstrap_rounds=1)
    k0_trace = np.array([2 ** 20 - k0["per_round"]] * n_rounds)

    print(f"loop: {summ['n_rounds']} rounds, all authenticated: {summ['auth_ok_all']}")
    print(f"  K per round {k[0]:.3e} bits, auth {a[0]:.1f}, seed {c[0]:.0f}, "
          f"u {u[0]:.3e} ({100 * summ['overhead']:.4f}% overhead)")
    print(f"  xi_MUI charged = {summ['xi_mui']:.4f} SNU, "
          f"rate = {summ['rate_per_symbol']:.4f} bits/symbol")
    print(f"  K_0 required = {k0['required']:.1f} bits of {k0['available']:.0f} "
          f"({100 * k0['fraction_of_k0']:.3f}%)")
    print(f"  tamper at round 8 -> aborted at {tampered['aborted_at']}, "
          f"reason: {tampered['reason']}")

    # --- plot -------------------------------------------------------------- #
    fig, axes = plotting.plt.subplots(1, 3, figsize=(13.4, 3.5))
    fig.subplots_adjust(wspace=0.38)

    ax = axes[0]
    # K_n and u_n differ by ~1e-5 relative: plot the *relative* budget instead
    ax.plot(rounds, 100.0 * (k - u) / k, "o-", color="#c0392b",
            label=r"$(a_n+c_n)/K_n$")
    ax.plot(rounds, 100.0 * a / k, "s--", color="#2471a3", label=r"$a_n/K_n$")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("round")
    ax.set_ylabel("overhead (% of $K_n$)")
    ax.set_title("Total overhead is 0.006% of $K_n$")
    ax.legend(fontsize=7)
    plotting.annotate(ax, f"$K_n$ = {k[0] / 1e6:.2f} Mbit/round\n"
                          f"absolute overhead = {a[0] + c[0]:.0f} bit",
                      xy=(0.32, 0.40), fontsize=7)

    ax = axes[1]
    ok_rounds = np.array([r.round_index for r in tampered["rounds"]])
    ax.bar(ok_rounds, [1.0] * len(ok_rounds), color="#1e8449", alpha=0.7,
           label="round completed")
    if tampered["aborted_at"]:
        ax.bar([tampered["aborted_at"]], [1.0], color="#c0392b",
               label="aborted (tamper detected)")
    ax.set_xlabel("round")
    ax.set_ylabel("outcome", labelpad=1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["", "", "done"])
    ax.set_title("Loop under MitM at round 8")
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[2]
    km = np.arange(5.0, 105.0, 2.5)
    for nmsg, col in zip((8, 128, 1024), ("#1e8449", "#d68910", "#c0392b")):
        need = [budget.k0_budget(bcfg, n_symbols,
                                 rate_at(cfg, float(L), xi_mui)["I_AB"],
                                 n_control_messages=nmsg,
                                 bits_per_message=1 << 20)["required"]
                for L in km]
        ax.plot(km, need, color=col, label=f"{nmsg} control msgs")
    ax.set_xlabel("fibre length (km)")
    ax.set_ylabel(r"$K_0$ required (bits)", labelpad=1)
    ax.set_title(r"Bootstrap cost: hundreds of bits, flat in $L$")
    ax.legend(fontsize=7)
    ax.ticklabel_format(axis="y", useOffset=False)

    plotting.save(fig, "e7_loop")

    return dict(summary=summ, rounds=rounds, K=k, a=a, c=c, u=u,
                k0=k0, k0_trace=k0_trace,
                tampered_aborted_at=tampered["aborted_at"],
                tampered_reason=tampered["reason"],
                n_symbols=n_symbols, distance_km=25.0)


if __name__ == "__main__":
    dump("e7_loop", main())
