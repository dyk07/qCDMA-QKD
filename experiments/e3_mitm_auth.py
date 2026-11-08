"""E3: The code-distribution gap -- MitM vs authentication.

Panel 1: survival probability of an unauthenticated vs authenticated MitM as a
         function of the number of rounds (tag length as a parameter).
Panel 2: what authentication *costs* -- bits per round for a realistic round,
         OTP vs eps-AXU, versus the round's key production.
Panel 3: minimum tag length needed to keep the per-round forgery probability
         below a target, as a function of how many messages a round carries.

Outputs
-------
results/figures/e3_mitm_auth.png
results/data/e3_mitm_auth.json
"""

from __future__ import annotations

import numpy as np

from _common import dump, header, rate_at, scenario

from qcdma_qkd import attack, budget, mui, plotting


def main() -> dict:
    header("E3  Code-distribution gap: MitM vs authenticated distribution")
    sc = scenario()
    cfg, bcfg, cdma = sc["cv"], sc["budget"], sc["cdma"]
    n_symbols = 10 ** 8
    symbol_rate = cdma.symbol_rate()   # per-user: chip_rate / L
    # same operating point as E5/E7: charge the MUI excess noise
    xi_mui = mui.required_xi(1e-12, mui.uniform_kappa(cdma.n_users - 1, 1.0),
                             cdma.spreading_factor)
    i_ab = rate_at(cfg, 25.0, xi_mui)["I_AB"]
    key_bits_round = rate_at(cfg, 25.0, xi_mui)["S_pos"] * n_symbols
    round_seconds = n_symbols / symbol_rate

    # --- 1. MitM survival ------------------------------------------------ #
    rounds = np.arange(0, 21)
    surv = {t: np.array([attack.detection_after_k_rounds(t, 8, int(n)) for n in rounds])
            for t in (32, 64, 128, 256)}

    # --- 2. authentication cost ------------------------------------------ #
    auth = {}
    for model in ("otp", "axu"):
        cfg_m = budget.BudgetConfig(**{**bcfg.__dict__, "mac_model": model})
        b = budget.round_budget(key_bits_round, n_symbols, i_ab, cfg_m)
        auth[model] = dict(a=b["a"], c=b["c"], u=b["u"], K=b["K"],
                           n_msg=b["n_messages"],
                           frac=(b["a"] + b["c"]) / b["K"])

    # --- 3. required tag length vs messages per round --------------------- #
    msgs = np.array([1, 8, 64, 512, 4096])
    t_req = {eps: np.ceil(np.log2(msgs) - np.log2(eps)) for eps in (1e-6, 1e-9, 1e-12)}

    print(f"round: {n_symbols:.0e} symbols, {round_seconds*1e3:.0f} ms, "
          f"K={key_bits_round:.3e} bits, {auth['axu']['n_msg']} messages")
    for m, v in auth.items():
        print(f"  {m}: auth={v['a']:.1f} bits ({100*v['frac']:.2e}% of K), "
              f"seed={v['c']:.0f} bits -> u={v['u']:.3e}")
    mu = attack.man_in_the_middle(128, authenticated=False)
    ma = attack.man_in_the_middle(128, authenticated=True)
    print(f"  MitM undetected (no auth): P={mu.detection_probability:.0f} -> "
          f"persists {mu.rounds_until_detection}")
    print(f"  MitM detected (auth):      P={ma.detection_probability:.12f}, "
          f"expected rounds to detect={ma.rounds_until_detection:.2f}")

    # --- plot ------------------------------------------------------------- #
    fig, axes = plotting.plt.subplots(1, 3, figsize=(12.6, 3.4))

    ax = axes[0]
    for t, c in zip((32, 64, 128, 256),
                    ("#c0392b", "#d68910", "#2471a3", "#1e8449")):
        ax.semilogy(rounds, surv[t], "-", color=c, label=f"tag {t} bit")
    ax.axhline(1.0, color="k", ls=":", lw=1, label="no authentication")
    ax.set_ylim(1e-300, 3)
    ax.set_xlabel("rounds since the attack started")
    ax.set_ylabel("P(attack still undetected)")
    ax.set_title("MitM survival, 8 messages/round")
    ax.legend(fontsize=7)

    ax = axes[1]
    labels = ["auth $a_n$", "seed $c_n$", "auth+seed"]
    x = np.arange(len(labels))
    w = 0.36
    for i, (model, col) in enumerate(zip(("otp", "axu"), ("#c0392b", "#2471a3"))):
        v = auth[model]
        vals = [v["a"], v["c"], v["a"] + v["c"]]
        ax.bar(x + (i - 0.5) * w, vals, w, color=col, alpha=0.85, label=model)
        for xi_, yi_ in zip(x + (i - 0.5) * w, vals):
            ax.annotate(f"{yi_:.3g}", (xi_, yi_), ha="center", va="bottom",
                        fontsize=6, rotation=90)
    ax.axhline(key_bits_round, color="k", ls="--", lw=1.2,
               label=r"$K_n$ (raw key)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("bits per round", labelpad=1)
    ax.set_title("Overhead vs raw key")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.set_ylim(1e-2, key_bits_round * 30)

    ax = axes[2]
    for eps, c in zip((1e-6, 1e-9, 1e-12), ("#1e8449", "#d68910", "#c0392b")):
        ax.plot(msgs, t_req[eps], "o-", color=c, label=f"eps={eps:.0e}")
    ax.set_xscale("log")
    ax.set_xlabel("authenticated messages per round")
    ax.set_ylabel("required tag length (bits)")
    ax.set_title("Tag length vs message count")
    ax.legend(fontsize=7)

    plotting.save(fig, "e3_mitm_auth")

    return dict(rounds=rounds, survival={str(k): v for k, v in surv.items()},
                auth=auth, msgs=msgs,
                tag_required={f"{k:.0e}": v for k, v in t_req.items()},
                n_symbols=n_symbols, key_bits_round=key_bits_round,
                mitm_unauthenticated=mu.note, mitm_authenticated=ma.note)


if __name__ == "__main__":
    dump("e3_mitm_auth", main())
