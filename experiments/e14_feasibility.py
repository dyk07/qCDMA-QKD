"""E14: the feasibility region, regenerated rather than transcribed.

Why this experiment exists
--------------------------
The epsilon-redistribution table in ``docs/10_parameters.md`` and in both paper
drafts was produced by hand at some earlier commit and never had a JSON source.
Re-deriving it exposed three separate defects:

1.  The quoted optimal ``L*`` values (1024, 1280, 1408, 1536, 1664) are all
    multiples of 128, i.e. a coarse scan grid, not an optimum.  On a 16-chip
    grid the true optima are 1456, 1712, 1856, 1984, 2128, and the table's
    numbers are 8% low at the far end.
2.  The table mixed routes: the rates were computed with the **exact**
    equal-power quantile while the claimed thresholds came from the
    **Laurent-Massart bound**.  The two differ by 27%.
3.  ``xi_crit`` was quoted as ``0.10312`` and used in the constraint
    ``xi + xi_mui < xi_crit``, which double counts the channel excess noise:
    0.10312 is already *total* excess noise minus the 0.01 channel term, i.e.
    an *allowance for the multiplexing term*.

Everything below is computed from the rate model in one pass, so the documents
and the papers can be checked against a file instead of against memory.

Two structural facts this experiment pins down
---------------------------------------------
*   ``xi_block_bound`` returns **0.0** when there is no interferer, because the
    Laurent-Massart inequality is stated for the sum over ``j`` and the
    ``2 a_max x*`` term needs a non-empty ``a``.  The lower bound on ``L`` is
    therefore **not** unconditional: it applies to ``K >= 1`` and scales with
    ``kappa_max``, not with ``L`` alone.  An earlier draft claimed the bound
    bound held at ``K = 0``; that was wrong and is retracted here.
*   ``K_crit ~ xi_crit L - 2 x*`` is very loose where it matters (3.9x at
    ``L = 1024``) because ``2 a_max x*`` with ``a_max = 1/L`` is not the
    dominant term once ``L`` is small relative to ``K``.  It is an
    order-of-magnitude formula and is labelled as such.

Outputs
-------
results/figures/e14_feasibility.png
results/data/e14_feasibility.json
"""

from __future__ import annotations

import math
import time

import numpy as np

from _common import dump, header, scenario

from qcdma_qkd import budget, cvqkd, mui, plotting

BLOCK_SYMBOLS = 10 ** 8
CHIP_RATE = 1.0e9

# Scan resolution for the spreading gain.  Multiplexing hardware uses powers of
# two, but an *optimum* must not be reported on a grid that is coarser than the
# curvature of the objective -- that is how the 128-multiple artefact arose.
L_SCAN = np.arange(256, 4097, 16)

# The two interference routes.  ``False`` is the exact chi-square quantile
# (equal power only); ``True`` is the Laurent-Massart theorem (any power).
ROUTES = (("exact", False), ("bound", True))


def _xi_of(eps_mui: float, L: int, route: str, n_users: int) -> float:
    """Multiplexing excess noise on the chosen route."""
    kappa = mui.uniform_kappa(n_users - 1, 1.0)
    if route == "bound":
        return float(mui.xi_block_bound(eps_mui, kappa, L))
    return float(mui.xi_block_exact(eps_mui, kappa, L))


def _rate(cfg, d_km: float, L: int, n_users: int, eps_round: float,
          route: str, n_symbols: int = BLOCK_SYMBOLS) -> dict:
    """Closed-loop rate at one operating point.

    All five epsilon components are set to ``eps_round`` so that a single knob
    controls the composition budget.  ``eps_pe`` is enabled: without it the
    reported rate is the nominal-parameter rate, which does not provision the
    confidence interval it claims.
    """
    T = cvqkd.total_transmittance(d_km, cfg.alpha_db_per_km, cfg.eta)
    xi = cfg.excess_noise + _xi_of(eps_round, L, route, n_users)
    return cvqkd.finite_size_rate(
        cfg.V_A, T, xi, cfg.eta, cfg.v_el, cfg.beta, cfg.detection,
        n_symbols=n_symbols, eps_sm=eps_round, eps_pa=eps_round,
        sample_fraction=0.1, eps_pe=eps_round)


def _allowance(cfg, d_km: float, n_symbols: int, tol: float = 1e-9) -> float:
    """Largest multiplexing excess noise that still leaves a positive rate.

    This is ``xi_crit_total - xi_channel``.  Defining it this way is what makes
    the constraint ``xi_mui <= allowance`` correct; writing
    ``xi + xi_mui <= xi_crit`` double counts ``xi``.
    """
    T = cvqkd.total_transmittance(d_km, cfg.alpha_db_per_km, cfg.eta)

    def alive(xi_total: float) -> float:
        return cvqkd.finite_size_rate(
            cfg.V_A, T, xi_total, cfg.eta, cfg.v_el, cfg.beta, cfg.detection,
            n_symbols=n_symbols, eps_sm=1e-10, eps_pa=1e-12,
            sample_fraction=0.1, eps_pe=1e-10)["S_pos"]

    lo, hi = 0.001, 2.0
    if alive(lo) <= 0.0:
        return 0.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if alive(mid) > 0.0 else (lo, mid)
    return lo - cfg.excess_noise


def _min_L(kappa, eps_mui: float, allowance: float) -> int | None:
    """Smallest ``L`` whose bound fits the allowance, or ``None``."""
    for L in L_SCAN:
        if float(mui.xi_block_bound(eps_mui, kappa, int(L))) <= allowance:
            return int(L)
    return None


def _max_M(kappa_unit, eps_mui: float, allowance: float, L: int) -> int | None:
    """Largest feasible user count at fixed ``L`` (exact, not the formula)."""
    for M in range(2, 512):
        kap = mui.uniform_kappa(M - 1, 1.0)
        if float(mui.xi_block_bound(eps_mui, kap, L)) > allowance:
            return M - 1
    return None


def main() -> dict:
    header("E14  Feasibility region, regenerated from the rate model")
    t0 = time.perf_counter()
    sc = scenario()
    cfg = sc["cv"]
    bcfg = sc["budget"]
    sec = sc["sec"]
    eps_mui = sec.eps_mui
    eps_target = sec.eps_total
    n = BLOCK_SYMBOLS

    # ------------------------------------------------------------------ #
    # 1. epsilon composition: which round counts are legal
    # ------------------------------------------------------------------ #
    eps_round = 3.05e-10                     # from e0 / security.from_config
    max_rounds = math.floor(eps_target / eps_round)
    rounds_rows = []
    for N in (1, 10, 100, 1000, 10_000, 100_000):
        need = eps_target / N
        rounds_rows.append(dict(rounds=N, eps_round_required=need))
    print(f"  eps_round = {eps_round:.3e} of {eps_target:.0e} -> at most "
          f"{max_rounds} rounds")

    # ------------------------------------------------------------------ #
    # 2. the cost of epsilon redistribution, both routes, converged L*
    # ------------------------------------------------------------------ #
    known_S = {}
    for route, _ in ROUTES:
        rows = []
        base = None
        for N in (1, 100, 1000, 10_000, 100_000):
            eps = eps_target / N
            best = max(((int(L), _rate(cfg, 25.0, int(L), 8, eps, route))
                        for L in L_SCAN),
                       key=lambda kv: kv[1]["S_pos"] * CHIP_RATE / kv[0])
            L_star = best[0]
            S_star = best[1]["S_pos"]
            bps = S_star * CHIP_RATE / L_star
            known_S[(route, N)] = (L_star, S_star, bps)
            if base is None:
                base = bps
            rows.append(dict(rounds=N, eps_round=eps, L_star=L_star,
                             S_star=S_star, bit_per_s=bps,
                             fraction_of_1_round=bps / base))
        print(f"  {route:5s} route: L* = "
              + ", ".join(f"{r['L_star']}" for r in rows))
        globals()[f"rows_{route}"] = rows

    # ------------------------------------------------------------------ #
    # 3. holder: L fixed at the default 1024, so no optimality claim
    # ------------------------------------------------------------------ #
    L_fixed = 1024
    fixed_rows = []
    for route, _ in ROUTES:
        base = None
        for N in (1, 10, 100, 1000, 10_000, 100_000):
            eps = eps_target / N
            S = _rate(cfg, 25.0, L_fixed, 8, eps, route)["S_pos"]
            bps = S * CHIP_RATE / L_fixed
            if base is None:
                base = bps
            fixed_rows.append(dict(route=route, rounds=N, eps_round=eps,
                                   S=S, bit_per_s=bps,
                                   fraction_of_1_round=bps / base))
    print(f"  L fixed at {L_fixed}: bound route falls from "
          f"{fixed_rows[6]['bit_per_s']:.0f} to {fixed_rows[11]['bit_per_s']:.0f} bit/s")

    # ------------------------------------------------------------------ #
    # 4. allowance, minimum L, and the retraction of the K=0 claim
    # ------------------------------------------------------------------ #
    allowance_rows = []
    for d in (10.0, 25.0, 40.0):
        for nn in (10 ** 7, 10 ** 8, 10 ** 9):
            allow = _allowance(cfg, d, nn)
            allowance_rows.append(dict(distance_km=d, n_symbols=nn,
                                       xi_allowance=allow))
    for r in allowance_rows:
        if r["n_symbols"] == 10 ** 8:
            print(f"  allowance d={r['distance_km']:.0f}km = "
                  f"{r['xi_allowance']:.5f}")

    kappa8 = mui.uniform_kappa(7, 1.0)
    lmin_rows = []
    for d in (10.0, 25.0, 40.0):
        for nn in (10 ** 7, 10 ** 8, 10 ** 9):
            allow = next(r["xi_allowance"] for r in allowance_rows
                         if r["distance_km"] == d and r["n_symbols"] == nn)
            lmin_rows.append(dict(
                distance_km=d, n_symbols=nn, xi_allowance=allow,
                L_min_bound=_min_L(kappa8, eps_mui, allow),
                L_min_closed_form=(2.0 * math.log(1.0 / eps_mui) / allow
                                   if allow > 0 else None)))
    for r in lmin_rows:
        print(f"  L_min d={r['distance_km']:.0f} n={r['n_symbols']:.0e}: "
              f"scan={r['L_min_bound']}  closed-form="
              f"{(r['L_min_closed_form'] or 0):.0f}")

    # The closed form 2 ln(1/eps)/xi is the three-term bound divided by
    # ignoring the first two terms; compare it with an actual scan where both
    # are present.
    floor_rows = []
    for L in (512, 1024, 1536, 2048):
        allow = next(r["xi_allowance"] for r in allowance_rows
                     if r["distance_km"] == 25.0 and r["n_symbols"] == n)
        bnd = float(mui.xi_block_bound(eps_mui, kappa8, L))
        floor_rows.append(dict(L=L, xi_bound=bnd, xi_allowance=allow,
                               fits=bnd <= allow))
    print("  bound at L=512..2048 (M=8, 25km): "
          + ", ".join(f"{r['xi_bound']:.4f}" for r in floor_rows))

    # K = 0: the bound is 0 by construction, so no unconditional floor exists.
    k0_rows = []
    for L in (128, 512, 1024, 4096):
        k0_rows.append(dict(L=L,
                            xi_bound_at_K0=float(
                                mui.xi_block_bound(eps_mui, np.array([]), L)),
                            xi_exact_at_K0=float(
                                mui.xi_block_exact(eps_mui, np.array([]), L))))
    print(f"  K=0: bound = {k0_rows[2]['xi_bound_at_K0']:.1f} "
          "-> the 2 a_max ln(1/eps) term needs K >= 1")

    # ------------------------------------------------------------------ #
    # 5. user ceiling: exact versus the loose closed form
    # ------------------------------------------------------------------ #
    allow25 = next(r["xi_allowance"] for r in allowance_rows
                   if r["distance_km"] == 25.0 and r["n_symbols"] == n)
    mcrit_rows = []
    for L in (1024, 1536, 2048, 4096):
        Mc = _max_M(kappa8, eps_mui, allow25, L)
        approx = allow25 * L - 2.0 * math.log(1.0 / eps_mui) + 1.0
        mcrit_rows.append(dict(L=L, M_crit_exact=Mc,
                               M_crit_closed_form=approx,
                               formula_overstatement=(approx / Mc
                                                      if Mc else None)))
    for r in mcrit_rows:
        print(f"  L={r['L']:5d}: M_crit exact={r['M_crit_exact']}, "
              f"formula={r['M_crit_closed_form']:.0f} "
              f"({r['formula_overstatement']:.1f}x loose)")

    # ------------------------------------------------------------------ #
    # 6. the headline operating point, gross versus net
    # ------------------------------------------------------------------ #
    xi_req = float(mui.required_xi(eps_mui, kappa8, 1024))
    r_head = cvqkd.finite_size_rate(
        cfg.V_A, cvqkd.total_transmittance(25.0, cfg.alpha_db_per_km, cfg.eta),
        cfg.excess_noise + xi_req, cfg.eta, cfg.v_el, cfg.beta, cfg.detection,
        n_symbols=n, eps_sm=sec.eps_sm, eps_pa=sec.eps_pa,
        sample_fraction=bcfg.pe_sample_fraction, eps_pe=sec.eps_qkd)
    b_head = budget.round_budget(r_head["S_pos"] * n, n, r_head["I_AB"], bcfg)
    Rsym = CHIP_RATE / 1024
    headline = dict(
        S=r_head["S_pos"], K_n=b_head["K"], u_n=b_head["u"],
        auth_bits=b_head["a"], seed_bits=b_head["c"],
        messages=b_head["n_messages"],
        gross_bit_per_s=r_head["S_pos"] * Rsym,
        net_bit_per_s=b_head["u"] * Rsym / n,
        symbol_rate=Rsym, round_seconds=n / Rsym,
        seed_fraction=b_head["c"] / b_head["K"],
        auth_fraction=b_head["a"] / b_head["K"])
    print(f"  headline S = {headline['S']:.6f}; gross = "
          f"{headline['gross_bit_per_s']:.1f} bit/s, net = "
          f"{headline['net_bit_per_s']:.1f} bit/s")
    print(f"  seed/K_n = {headline['seed_fraction']:.3e}, "
          f"auth/K_n = {headline['auth_fraction']:.3e}")

    # ------------------------------------------------------------------ #
    # 7. reach
    # ------------------------------------------------------------------ #
    reach_rows = []
    for d in np.arange(38.0, 44.0, 0.5):
        rr = cvqkd.finite_size_rate(
            cfg.V_A,
            cvqkd.total_transmittance(float(d), cfg.alpha_db_per_km, cfg.eta),
            cfg.excess_noise + xi_req, cfg.eta, cfg.v_el, cfg.beta,
            cfg.detection, n_symbols=n, eps_sm=sec.eps_sm, eps_pa=sec.eps_pa,
            sample_fraction=bcfg.pe_sample_fraction, eps_pe=sec.eps_qkd)
        bb = budget.round_budget(rr["S_pos"] * n, n, rr["I_AB"], bcfg)
        reach_rows.append(dict(distance_km=float(d), S=rr["S_pos"],
                               K_n=bb["K"], u_n=bb["u"],
                               feasible=bool(bb["u"] >= 0.0)))
    last_ok = max((r["distance_km"] for r in reach_rows if r["feasible"]),
                  default=0.0)
    print(f"  budget-limited reach = {last_ok:.1f} km")

    # ------------------------------------------------------------------ #
    # figure
    # ------------------------------------------------------------------ #
    fig, axes = plotting.plt.subplots(1, 2, figsize=(9.0, 3.4))
    ax = axes[0]
    for route, style in (("exact", "o-"), ("bound", "s--")):
        rows = [r for r in fixed_rows if r["route"] == route]
        ax.semilogx([r["rounds"] for r in rows],
                    [r["bit_per_s"] for r in rows], style,
                    label=f"{route} route")
    ax.set_xlabel("rounds composed under the same $\\varepsilon$ target")
    ax.set_ylabel("bit/s per user")
    ax.set_title(f"$\\varepsilon$ redistribution (L={L_fixed})")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot([r["L"] for r in floor_rows],
            [r["xi_bound"] for r in floor_rows], "o-",
            label="Laurent--Massart bound")
    ax.axhline(allow25, color="crimson", ls=":",
               label="allowance at 25 km")
    ax.set_xlabel("spreading gain $L$")
    ax.set_ylabel("$\\xi$ budget")
    ax.set_title("bound vs allowance")
    ax.legend(fontsize=7)
    plotting.save(fig, "e14_feasibility")

    out = dict(
        eps_round=eps_round, eps_target=eps_target, max_rounds=max_rounds,
        rounds=rounds_rows,
        eps_penalty_optimised={route: globals()[f"rows_{route}"]
                               for route, _ in ROUTES},
        eps_penalty_fixed_L=fixed_rows,
        allowance=allowance_rows, L_min=lmin_rows, floor=floor_rows,
        k_zero=k0_rows, M_crit=mcrit_rows, headline=headline,
        reach=reach_rows, reach_km=last_ok,
        note=("the lower bound on L holds for K>=1 and scales with kappa_max; "
              "at K=0 the Laurent-Massart bound is identically zero"),
        runtime_s=time.perf_counter() - t0)
    print(f"  runtime {out['runtime_s']:.1f}s")
    return out


if __name__ == "__main__":
    dump("e14_feasibility", main())
