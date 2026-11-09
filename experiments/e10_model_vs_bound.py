"""E10: every model in the project, next to a provable bound.

The rest of the experiments answer "what is the number?".  This one answers
"how much of that number rests on an assumption?" -- for each ingredient it
reports

* the **model** value actually used by the pipeline,
* the **rigorous bound** (a theorem, or an exact distribution where available),
* the **gap**, and
* which direction the model errs in (optimistic = overstates the key rate).

The output feeds ``docs/09_model_vs_bound.md``; the numbers in that document are
checked against ``results/data/e10_model_vs_bound.json`` by
``tools/doc_check.py``.

Outputs
-------
results/figures/e10_model_vs_bound.png
results/data/e10_model_vs_bound.json
"""

from __future__ import annotations

import numpy as np

from _common import dump, header, scenario

from qcdma_qkd import attack, cvqkd, mui, plotting

N_SYMBOLS = 10 ** 8
DISTANCE_KM = 25.0
EPS_PE = 1e-10
EPS_ENTROPY = 1e-12


def main() -> dict:
    header("E10  Model vs bound: how much of the result is assumed?")
    sc = scenario()
    cfg, cdma, bcfg = sc["cv"], sc["cdma"], sc["budget"]
    L = cdma.spreading_factor
    kappa = mui.uniform_kappa(cdma.n_users - 1, 1.0)
    T = cvqkd.total_transmittance(DISTANCE_KM, cfg.alpha_db_per_km, cfg.eta)

    out: dict = dict(distance_km=DISTANCE_KM, n_symbols=N_SYMBOLS,
                     L=L, n_interferers=int(kappa.size))

    # ------------------------------------------------------------------ #
    # 1. MUI excess noise: exact quantile vs Gaussian fit vs Laurent-Massart
    # ------------------------------------------------------------------ #
    mui_rows = []
    for eps in (1e-6, 1e-9, 1e-12, 1e-15):
        d = mui.xi_gap_vs_bound(eps, kappa, L)
        d["eps_mui"] = eps
        mui_rows.append(d)
    out["mui"] = mui_rows
    xi_used = mui.required_xi(1e-12, kappa, L)
    out["xi_mui_used"] = float(xi_used)
    out["xi_mui_bound"] = float(mui.xi_block_bound(1e-12, kappa, L))

    print(f"  MUI at eps=1e-12: model(gamma)={xi_used:.5f}  "
          f"gaussian={mui.required_xi_normal(1e-12, kappa, L):.5f}  "
          f"bound={out['xi_mui_bound']:.5f}")

    # ------------------------------------------------------------------ #
    # 2. Seed entropy: mean leakage vs binomial upper tail
    # ------------------------------------------------------------------ #
    ent_rows = []
    for p in (0.05, 0.10, 0.20):
        d = attack.entropy_gap_vs_tail(bcfg.seed_bits, L, p, EPS_ENTROPY)
        d["p_chip"] = p
        d["k_mean"] = p * L
        d["k_star"] = bcfg.seed_bits - d["entropy_tail"]
        ent_rows.append(d)
    out["entropy"] = ent_rows
    out["seed_bits"] = int(bcfg.seed_bits)
    out["eps_entropy"] = EPS_ENTROPY

    # ------------------------------------------------------------------ #
    # 3. Refresh constraint: legacy (tail only) vs rigorous (tail + guessing)
    # ------------------------------------------------------------------ #
    refresh_rows = []
    for eps in (1e-6, 1e-9, 1e-12, 1e-15):
        c = attack.refresh_constraint_tail(bcfg.seed_bits, L, 1.0, eps)
        refresh_rows.append(dict(eps_code=eps,
                                 f_min_rigorous=c["f_min"],
                                 f_min_legacy=c.get("f_min_legacy", np.nan),
                                 f_min_mean=c.get("f_min_mean", np.nan),
                                 k_star=int(c["k_star"]),
                                 p_exceed=c["p_exceed"],
                                 p_guess=c["p_guess"],
                                 eps_total=c["violation"]))
    out["refresh"] = refresh_rows
    print(f"  refresh: f_min rigorous={refresh_rows[2]['f_min_rigorous']:.5f} "
          f"(tail-only was {refresh_rows[2]['f_min_legacy']:.5f})")

    # ------------------------------------------------------------------ #
    # 4. PE confidence intervals: Gaussian vs exact chi-square
    # ------------------------------------------------------------------ #
    pe_rows = []
    for n_tot in (10 ** 5, 10 ** 6, 10 ** 7, 10 ** 8):
        d = cvqkd.pe_gap_vs_exact(T, xi_used, int(0.1 * n_tot), EPS_PE,
                                  cfg.eta, cfg.V_A, cfg.v_el)
        d["n_pe"] = int(0.1 * n_tot)
        pe_rows.append(d)
    out["pe"] = pe_rows
    worst_xi = pe_rows[0]
    print(f"  PE: exact xi_ub={worst_xi['xi_ub_exact']:.5f} vs gaussian "
          f"{worst_xi['xi_ub_gaussian']:.5f} at n_pe={worst_xi['n_pe']:.0e}")

    # ------------------------------------------------------------------ #
    # 5. Rate: nominal parameters vs PE confidence bounds
    # ------------------------------------------------------------------ #
    nominal = cvqkd.finite_size_rate(
        cfg.V_A, T, xi_used, cfg.eta, cfg.v_el, cfg.beta, cfg.detection,
        n_symbols=N_SYMBOLS, eps_sm=sc["sec"].eps_sm, eps_pa=sc["sec"].eps_pa,
        sample_fraction=bcfg.pe_sample_fraction)
    with_pe = cvqkd.finite_size_rate(
        cfg.V_A, T, xi_used, cfg.eta, cfg.v_el, cfg.beta, cfg.detection,
        n_symbols=N_SYMBOLS, eps_sm=sc["sec"].eps_sm, eps_pa=sc["sec"].eps_pa,
        sample_fraction=bcfg.pe_sample_fraction, eps_pe=EPS_PE)
    asym = cvqkd.asymptotic_rate(cfg.V_A, T, xi_used, cfg.eta, cfg.v_el,
                                 cfg.beta, cfg.detection)
    out["rate"] = dict(
        asymptotic=float(asym["S_pos"]),
        finite_nominal=float(nominal["S_pos"]),
        finite_with_pe=float(with_pe["S_pos"]),
        T_used_nominal=float(nominal["T_used"]),
        xi_used_nominal=float(nominal["xi_used"]),
        T_used_with_pe=float(with_pe["T_used"]),
        xi_used_with_pe=float(with_pe["xi_used"]),
        finite_over_asymptotic=float(nominal["S_pos"] / asym["S_pos"]),
        pe_over_finite=float(with_pe["S_pos"] / nominal["S_pos"]),
    )
    r = out["rate"]
    print(f"  rate @{DISTANCE_KM:.0f} km: asym={r['asymptotic']:.5f} -> "
          f"finite={r['finite_nominal']:.5f} ({r['finite_over_asymptotic']:.1%})"
          f" -> with PE bounds={r['finite_with_pe']:.5f} ({r['pe_over_finite']:.1%})")

    # ------------------------------------------------------------------ #
    # 6. Code layer: computational PRF claim (no numeric bound exists)
    # ------------------------------------------------------------------ #
    out["prf"] = dict(
        prf_advantage_128bit=attack.residual_seed_entropy(
            bcfg.seed_bits, L, 0.0, "prf"),
        distinguishing_advantage_1e9_queries=(
            2.0 ** (-bcfg.seed_bits / 2.0)
            + 1e9 * 2.0 ** (-bcfg.seed_bits)),
        note="toy bound only: no proof that BLAKE2b is a PRF",
    )
    print(f"  PRF: toy distinguishing advantage (1e9 queries) = "
          f"{out['prf']['distinguishing_advantage_1e9_queries']:.2e}  "
          f"[ASSUMPTION, not a theorem]")

    # ------------------------------------------------------------------ #
    # figure
    # ------------------------------------------------------------------ #
    fig, axes = plotting.plt.subplots(1, 3, figsize=(13.2, 3.5))

    ax = axes[0]
    eps_grid = np.array([r["eps_mui"] for r in mui_rows])
    ax.semilogx(eps_grid, [r["xi_exact"] for r in mui_rows], "o-",
                color="#1e8449", label="exact quantile")
    ax.semilogx(eps_grid, [r["xi_gaussian"] for r in mui_rows], "s--",
                color="#c0392b", label="Gaussian fit (model)")
    ax.semilogx(eps_grid, [r["xi_bound"] for r in mui_rows], "^:",
                color="#2471a3", label="Laurent-Massart bound")
    ax.set_xlabel(r"$\epsilon_{\mathrm{MUI}}$")
    ax.set_ylabel(r"$\xi_{\mathrm{MUI}}$ (SNU)")
    ax.set_title("MUI: model vs rigorous bound")
    ax.legend(fontsize=7)
    ax.invert_xaxis()

    ax = axes[1]
    names = ["asymptotic", "finite\n(nominal)", "finite\n(PE bounds)"]
    vals = [r["asymptotic"], r["finite_nominal"], r["finite_with_pe"]]
    ax.bar(names, vals, color=["#7f8c8d", "#2471a3", "#c0392b"], alpha=0.85)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.02, f"{v:.4f}", ha="center", fontsize=7)
    ax.set_ylabel("S (bits/symbol)")
    ax.set_title(f"Rate @ {DISTANCE_KM:.0f} km: each step is an assumption")
    ax.tick_params(labelsize=7)

    ax = axes[2]
    p_grid = [r["p_chip"] for r in ent_rows]
    ax.plot(p_grid, [r["entropy_mean"] for r in ent_rows], "s--",
            color="#c0392b", label="mean leakage (model)")
    ax.plot(p_grid, [r["entropy_tail"] for r in ent_rows], "o-",
            color="#1e8449", label=r"upper tail ($10^{-12}$)")
    ax.set_xlabel(r"$p_{\mathrm{blind}}$")
    ax.set_ylabel(r"$H_{\min}$ of the seed (bit)")
    ax.set_title("Seed entropy: mean vs worst case")
    ax.legend(fontsize=7)

    plotting.save(fig, "e10_model_vs_bound")
    return out


if __name__ == "__main__":
    dump("e10_model_vs_bound", main())
