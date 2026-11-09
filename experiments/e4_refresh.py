"""E4: Cross-round leakage -- the refresh / seed / spreading-factor constraint.

This experiment makes the *price of information-theoretic code secrecy* explicit.
The sample model does not pay it (it uses a computational PRF with f = 1), so
every panel also shows where that choice sits.

Panel 1: min-entropy of the *current* code seed after one round of leakage vs
         the refresh fraction, for three seed lengths (extractor model).  The
         dotted lines are the binomial-tail design constraint f >= f_min with
         k_req = log2(1/eps_code); the PRF route needs no refreshing at all.
Panel 2: eps_code vs refresh fraction on a log scale, extractor vs PRF, with the
         eps_code target marked.  The gap between the two curves is the price.
Panel 3: f_min over the (L, seed) grid; 0 means the constraint does not bind.

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
    cdma = sc["cdma"]
    L = cdma.spreading_factor
    seed_cfg = sc["budget"].seed_bits

    f_grid = np.linspace(0.0, 1.0, 101)
    seeds = (128, 256, 512)

    # --- 1/2. seed min-entropy and eps_code ------------------------------- #
    # NOTE: the plotted H_min curve is the *mean* trajectory E[H_min]; the
    # actual quantity is random (binomial k_known), so the design threshold uses
    # the binomial upper tail, not this curve.
    h = {s: np.array([attack.residual_seed_entropy(s, L, (1 - f) * 1.0) for f in f_grid])
         for s in seeds}
    eps = {s: np.array([attack.cross_round_eps(f, s, L, 1.0) for f in f_grid])
           for s in seeds}
    eps_prf = 2.0 ** (-256)
    eps_target = 1e-12
    k_req = attack.required_entropy_bits(eps_target)

    # mean closed form (kept for monotonicity / analytic reference)
    constraint = {s: attack.refresh_security_constraint(s, L, eps_code=eps_target)
                  for s in seeds}
    # DEFAULT: exact binomial upper tail on k_known
    tail = {s: attack.refresh_constraint_tail(s, L, 1.0, eps_code=eps_target)
            for s in seeds}
    for s in seeds:
        c, t = constraint[s], tail[s]
        print(f"seed={s:4d} bits, L={L}: f_min mean={c['f_min']:.4f} "
              f"tail={t['f_min']:.4f} "
              f"(k_known <= {c['max_known_chips']:.1f}, k_req = {k_req:.1f} bit)")

    # --- 3. feasible region ----------------------------------------------- #
    L_grid = np.array([128, 256, 512, 1024, 2048, 4096])
    seed_grid = np.array([128, 256, 512, 1024])
    feasible = np.zeros((len(seed_grid), len(L_grid)))
    for i, s in enumerate(seed_grid):
        for j, LL in enumerate(L_grid):
            feasible[i, j] = attack.refresh_constraint_tail(
                s, int(LL), 1.0, eps_code=eps_target)["f_min"]

    # --- where do we sit by default? -------------------------------------- #
    d, dt = constraint[256], tail[256]
    print(f"default: seed=256, L={L} -> f_min(mean)={d['f_min']:.4f}, "
          f"f_min(tail)={dt['f_min']:.4f}")
    print(f"  at the tail threshold n_reused = "
          f"{int(round((1.0 - dt['f_min']) * L))} chips, "
          f"so k_known must stay <= {d['max_known_chips']:.1f}")
    print("  old H_min>0 rule said 0.7500, but eps_code there = 1.0 (total loss)")

    # --- what the sample model actually does ------------------------------- #
    chosen = cdma.refresh_constraint(seed_cfg, eps_code=eps_target)
    print(f"\n  SAMPLE MODEL: code_model={chosen['code_model']}, "
          f"f={chosen['refresh_fraction']:g}, "
          f"f_min={chosen['f_min']:.4f} -> satisfied={chosen['satisfied']}")
    print(f"  security: {chosen['security']}")
    if chosen["code_model"] == "prf":
        print(f"  => the constraint does NOT bind; f=1 additionally kills "
              f"cross-round correlation")
        print(f"  => the extractor route would require f >= "
              f"{tail[seed_cfg]['f_min']:.4f} -- that is the IT price")

    # --- plot -------------------------------------------------------------- #
    fig, axes = plotting.plt.subplots(1, 3, figsize=(13.6, 3.6))
    fig.subplots_adjust(wspace=0.42)

    ax = axes[0]
    for s, c in zip(seeds, ("#c0392b", "#2471a3", "#1e8449")):
        ax.plot(f_grid, h[s], color=c,
                label="seed %d bit (mean $H_{\\min}$)" % s)
        ax.axvline(tail[s]["f_min"], color=c, ls=":", lw=1)
    # the security margin is unspendable, so H_min must stay above k_req
    ax.axhline(k_req, color="k", ls="-.", lw=1.2,
               label="$k_{\\rm req}$ = %.1f bit" % k_req)
    # shade only the region where *every* seed is exposed
    ax.axvspan(0.0, min(tail[s]["f_min"] for s in seeds),
               color="#c0392b", alpha=0.10,
               label="all seeds exposed")
    ax.set_xlabel("refresh fraction f")
    ax.set_ylabel(r"$H_{\min}(c_n)$ (bits)")
    ax.set_title("Mean seed entropy after one leaked\nround (dotted = tail $f_{\\min}$)")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_ylim(-20, 610)

    ax = axes[1]
    for s, c in zip(seeds, ("#c0392b", "#2471a3", "#1e8449")):
        ax.semilogy(f_grid, np.clip(eps[s], 1e-300, 1), color=c,
                    label=f"extractor, {s} bit")
    ax.axhline(eps_target, color="#8e44ad", ls="-.", lw=1.2,
               label=fr"target $\epsilon$ = {eps_target:g}")
    ax.axhline(eps_prf, color="k", ls="--", lw=1.2, label="computational PRF")
    ax.set_xlabel("refresh fraction f")
    ax.set_ylabel(r"$\epsilon_{\mathrm{code}}$", labelpad=1)
    ax.set_title("Extractor vs PRF: the price of\ninformation-theoretic codes")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_ylim(1e-150, 3)

    ax = axes[2]
    im = ax.imshow(feasible, origin="lower", aspect="auto", cmap="viridis",
                   vmin=0, vmax=1)
    ax.set_xticks(range(len(L_grid)))
    ax.set_xticklabels(L_grid)
    ax.set_yticks(range(len(seed_grid)))
    ax.set_yticklabels(seed_grid)
    ax.set_xlabel("spreading factor L")
    ax.set_ylabel("seed length (bits)")
    ax.set_title(r"required refresh $f_{\min}$ (extractor)" + "\n"
                 + fr"at $\epsilon_{{\rm code}}$ = {eps_target:g}")
    for i in range(len(seed_grid)):
        for j in range(len(L_grid)):
            v = feasible[i, j]
            label = "0" if v <= 0.0 else f"{v:.2f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=7,
                    color="w" if v < 0.55 else "k")
    fig.colorbar(im, ax=ax, fraction=0.046)

    plotting.save(fig, "e4_refresh")

    return dict(f_grid=f_grid, L=L, seeds=list(seeds),
                h_min={str(s): h[s] for s in seeds},
                eps_code={str(s): np.clip(eps[s], 1e-300, 1) for s in seeds},
                eps_prf=eps_prf, eps_target=eps_target, k_req=k_req,
                tail={str(s): c for s, c in tail.items()},
                chosen_model=chosen["code_model"],
                chosen_refresh=chosen["refresh_fraction"],
                chosen_security=chosen["security"],
                chosen_f_min=chosen["f_min"],
                constraint={str(s): c for s, c in constraint.items()},
                L_grid=L_grid, seed_grid=seed_grid, f_min_map=feasible)


if __name__ == "__main__":
    dump("e4_refresh", main())
