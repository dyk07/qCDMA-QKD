"""E13: How much does the equal-interferer-power assumption buy?

Why this exists
---------------
Every headline number in this project uses ``kappa_j = 1`` for all ``j`` (equal
interferer powers).  That is not a cosmetic convenience: for equal powers the
block-averaged MUI ``xi_block = (1/L) sum_j kappa_j z_j^2`` is exactly
``(kappa/L) * chi^2_K``, so its ``eps``-quantile is a **closed form**.  For
unequal powers there is no closed form, and ``mui.xi_block_exact`` silently
falls back to a moment-matched Gamma, which is **not** the true quantile.  Which
way the error goes decides whether the headline numbers are safe, so it has to
be measured rather than assumed.

The quantity that matters is

    nu = (sum_j kappa_j)^2 / sum_j kappa_j^2          (effective interferers)

with ``1 <= nu <= K``.  It is exact that

    rel.std[xi_block] = sqrt(2 / nu)          (verified below)

so ``nu`` is the single number that says how concentrated the interference is.
``nu = K`` means equal powers; ``nu -> 1`` means one interferer dominates.

What the experiment measures
----------------------------
1. ``rel.std == sqrt(2/nu)`` exactly, and the Gamma route's quantile error
   against a **sampling-free** reference (saddlepoint), not against Monte Carlo.
   Monte Carlo was tried first and cannot be trusted here: at ``eps = 1e-6`` it
   produces about two tail events per two million draws, so its "quantile" is
   the largest order statistic, which scattered by ~13% run to run and once
   produced a spurious non-monotone error curve.
2. Rate versus concentration at 25 km, under **both** routes: the Gamma
   moment-match (tight but biased) and the Laurent--Massart bound (a theorem,
   valid for any ``kappa``, loose).
3. Where the 25 km link dies.
4. The **false-security gap**: what a link designed for equal powers silently
   assumes when the interference is in fact concentrated.

The honest headline
-------------------
At 25 km the MUI budget is so tight that the link survives **only** in a narrow
band around equal powers.  The equal-power assumption is therefore a
**load-bearing design assumption**, not a convenience -- and the analysis here
is what turns that sentence into a number.

Outputs
-------
results/figures/e13_kappa_concentration.png
results/data/e13_kappa_concentration.json

Runtime: a few seconds (deliberately: earlier drafts used a Python-level Imhof
integral that took forever; everything here is closed-form, scalar, or a short
saddlepoint bisection).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

from _common import dump, header, scenario

from qcdma_qkd import cvqkd, mui, plotting

# Keep the whole experiment inside a hard budget so a regression in the numerics
# cannot silently turn into an overnight run (this project has done that twice).
BUDGET_S = 300.0
EPS_MUI = 1e-12        # per-round composition parameter (the design value)
EPS_REF = 1e-6         # quantile at which the routes are compared
MC_DRAWS = 2_000_000   # only used to *corroborate* at eps=1e-4, never at 1e-6


def geometric_kappa(n_interferers: int, ratio: float) -> np.ndarray:
    """Powers decaying geometrically, normalised to the *same total* power.

    ``ratio = 1`` is the equal-power default; ``ratio -> 0`` is one dominant
    interferer.  Normalising ``sum(kappa) == K`` keeps the **mean** MUI fixed,
    so any change in the rate is attributable to concentration alone, not to
    the interferers becoming louder.
    """
    weights = np.power(float(ratio), np.arange(n_interferers))
    return n_interferers * weights / weights.sum()


def effective_nu(kappa: np.ndarray) -> float:
    """DoF of the equivalent chi-square: (sum k)^2 / sum k^2."""
    return float(np.sum(kappa) ** 2 / np.sum(kappa ** 2))


def main() -> dict:
    t0 = time.perf_counter()

    def check(tag: str) -> None:
        elapsed = time.perf_counter() - t0
        if elapsed > BUDGET_S:
            raise RuntimeError(f"E13 exceeded {BUDGET_S}s at {tag} ({elapsed:.1f}s)")

    def reference_quantile(kappa: np.ndarray, eps: float) -> float:
        """Sampling-free reference quantile (see mui.xi_block_quantile_saddle)."""
        return float(mui.xi_block_quantile_saddle(eps, kappa, L))

    header("E13  Interferer power imbalance: is 'equal kappa' load-bearing?")
    sc = scenario()
    cfg, cdma, bcfg, sec = sc["cv"], sc["cdma"], sc["budget"], sc["sec"]
    L = int(cdma.spreading_factor)
    K = int(cdma.n_users) - 1
    n_symbols = 10 ** 8
    distance_km = 25.0
    T = cvqkd.total_transmittance(distance_km, cfg.alpha_db_per_km, cfg.eta)

    def rate(xi_extra: float) -> float:
        return cvqkd.finite_size_rate(
            cfg.V_A, T, cfg.excess_noise + xi_extra, cfg.eta, cfg.v_el,
            cfg.beta, cfg.detection, n_symbols=n_symbols,
            eps_pe=sec.eps_qkd, sample_fraction=bcfg.pe_sample_fraction)["S_pos"]

    ratio_grid = (1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1, 1e-4)

    # ------------------------------------------------------------------ #
    # 0. Calibrate the reference itself, then corroborate with Monte Carlo
    # ------------------------------------------------------------------ #
    # The reference is only useful if it reproduces the one case we *do* know
    # exactly: equal powers, where xi_block is chi^2_K / L.  That check runs
    # first so a broken reference cannot quietly become the baseline for
    # everything downstream.
    calib_rows = []
    for eps in (1e-4, 1e-6, 1e-9, 1e-12):
        kappa = geometric_kappa(K, 1.0)
        exact = float(mui.xi_block_exact(eps, kappa, L))
        ref = float(mui.xi_block_quantile_saddle(eps, kappa, L))
        calib_rows.append(dict(eps=eps, exact=exact, reference=ref,
                               rel_error=ref / exact - 1.0))
    print("  reference calibration (equal power, exact = chi2_K/L): "
          + ", ".join(f"{100 * r['rel_error']:+.2f}%@{r['eps']:g}"
                      for r in calib_rows))

    rng = np.random.default_rng(0)
    mc_rows = []
    for ratio in (1.0, 0.8, 0.5, 0.3):
        kappa = geometric_kappa(K, ratio)
        z = rng.normal(size=(MC_DRAWS, K))
        for eps in (1e-4,):        # the only eps MC resolves with 2e6 draws
            mc = float(np.quantile((kappa * z ** 2).sum(axis=1) / L, 1.0 - eps))
            mc_rows.append(dict(ratio=ratio, eps=eps, mc=mc,
                                reference=reference_quantile(kappa, eps),
                                mc_is_above_bound=bool(
                                    mc <= mui.xi_block_bound(eps, kappa, L))))
        check("mc-corroboration")
    worst_mc = max(mc_rows, key=lambda r: abs(r["mc"] / r["reference"] - 1.0))
    print(f"  Monte Carlo corroboration at eps=1e-4: reference vs MC agrees "
          f"within {100 * abs(worst_mc['mc'] / worst_mc['reference'] - 1.0):.2f}% "
          f"(worst at ratio={worst_mc['ratio']:g})")

    # ------------------------------------------------------------------ #
    # 1. rel.std == sqrt(2/nu) exactly, and the Gamma route's quantile
    #    error against the calibrated reference
    # ------------------------------------------------------------------ #
    identity_rows = []
    for ratio in (1.0, 0.8, 0.5, 0.3, 0.1):
        kappa = geometric_kappa(K, ratio)
        nu = effective_nu(kappa)
        rel_std = mui.xi_block_std(kappa, L) / mui.xi_block_mean(kappa, L)
        pred = float(np.sqrt(2.0 / nu))
        ref_q = reference_quantile(kappa, EPS_REF)
        gam_q = float(mui.xi_block_exact(EPS_REF, kappa, L))
        bnd_q = float(mui.xi_block_bound(EPS_REF, kappa, L))
        identity_rows.append(dict(
            ratio=ratio, nu=nu, n_interferers=K,
            rel_std=rel_std, rel_std_predicted=pred,
            rel_std_identity_error=abs(rel_std - pred),
            xi_quantile_reference=ref_q, xi_quantile_gamma=gam_q,
            xi_quantile_bound=bnd_q,
            gamma_error=gam_q / ref_q - 1.0,
            bound_error=bnd_q / ref_q - 1.0,
        ))
        check("identity")
    print(f"  rel.std identity: max |sqrt(2/nu) - measured| = "
          f"{max(r['rel_std_identity_error'] for r in identity_rows):.2e}")
    for r in identity_rows:
        print(f"    ratio={r['ratio']:g}: nu={r['nu']:.4f} "
              f"gamma{100 * r['gamma_error']:+7.2f}%  "
              f"bound{100 * r['bound_error']:+7.2f}%  (vs reference)")
    print(f"  Laurent-Massart is >= reference everywhere: "
          f"{all(r['bound_error'] >= -1e-9 for r in identity_rows)}")

    # ------------------------------------------------------------------ #
    # 2. Rate vs concentration, both routes
    # ------------------------------------------------------------------ #
    rate_rows = []
    for ratio in ratio_grid:
        kappa = geometric_kappa(K, ratio)
        nu = effective_nu(kappa)
        xi_gam = float(mui.xi_block_exact(EPS_MUI, kappa, L))
        xi_bnd = float(mui.xi_block_bound(EPS_MUI, kappa, L))
        rate_rows.append(dict(
            ratio=ratio, nu=nu,
            rel_std=mui.xi_block_std(kappa, L) / mui.xi_block_mean(kappa, L),
            xi_quantile_gamma=xi_gam, xi_bound=xi_bnd,
            rate_gamma=rate(xi_gam), rate_bound=rate(xi_bnd),
            bit_per_s_gamma=rate(xi_gam) * cdma.symbol_rate(),
            bit_per_s_bound=rate(xi_bnd) * cdma.symbol_rate(),
        ))
        check("rate")
    base = rate_rows[0]["bit_per_s_bound"]
    for r in rate_rows:
        r["fraction_of_equal_bound"] = (r["bit_per_s_bound"] / base
                                        if base > 0 else 0.0)
        r["fraction_of_equal_gamma"] = (
            r["bit_per_s_gamma"] / rate_rows[0]["bit_per_s_gamma"]
            if rate_rows[0]["bit_per_s_gamma"] > 0 else 0.0)
    print(f"  equal power: S={rate_rows[0]['rate_gamma']:.6f} (Gamma), "
          f"S={rate_rows[0]['rate_bound']:.6f} (bound)")

    # ------------------------------------------------------------------ #
    # 3. How far from equal power can the 25 km link drift?
    # ------------------------------------------------------------------ #
    # Bisection invariant: ``alive(lo)`` is True and ``alive(hi)`` is False.
    # ``ratio = 1`` (equal) is the alive end, ``ratio -> 0`` (one dominant
    # interferer) is the dead end.  Getting these two the wrong way round
    # silently returns the dead end as the answer, so it is asserted.
    def alive(ratio: float) -> bool:
        kappa = geometric_kappa(K, ratio)
        return rate(float(mui.xi_block_bound(EPS_MUI, kappa, L))) > 0.0

    assert alive(1.0), "the baseline equal-power design must be feasible"
    assert not alive(1e-6), "full concentration must not be feasible at 25 km"
    lo, hi = 1e-6, 1.0                  # lo = dead, hi = alive
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if alive(mid):
            hi = mid
        else:
            lo = mid
    ratio_star = hi                     # smallest feasible ratio
    nu_star = effective_nu(geometric_kappa(K, ratio_star))
    check("threshold")
    print(f"  feasible for ratio >= {ratio_star:.4f} "
          f"(nu = {nu_star:.4f}); below that xi_MUI(1e-12) kills the rate")

    # ------------------------------------------------------------------ #
    # 4. False-security gap
    # ------------------------------------------------------------------ #
    xi_design = rate_rows[0]["xi_quantile_gamma"]
    false_rows = []
    for ratio in (0.8, 0.5, 0.3):
        kappa = geometric_kappa(K, ratio)
        demand = float(mui.xi_block_bound(EPS_MUI, kappa, L))
        false_rows.append(dict(
            ratio=ratio, nu=effective_nu(kappa),
            xi_design=xi_design, xi_actually_required=demand,
            understatement=demand / xi_design - 1.0,
            rate_if_correctly_provisioned=rate(demand),
        ))
        check("false-security")
    for r in false_rows:
        print(f"  ratio={r['ratio']:g}: design {r['xi_design']:.4f} but need "
              f">= {r['xi_actually_required']:.4f} "
              f"({100 * r['understatement']:+.0f}%)")

    # ------------------------------------------------------------------ #
    # figure
    # ------------------------------------------------------------------ #
    fig, axes = plotting.plt.subplots(1, 3, figsize=(12.6, 3.4))

    ax = axes[0]
    nus = [r["nu"] for r in rate_rows]
    ax.plot(nus, [r["rel_std"] for r in rate_rows], "o-", color="#2471a3",
            label="measured")
    grid = np.linspace(1.0, K, 200)
    ax.plot(grid, np.sqrt(2.0 / grid), "--", color="#c0392b",
            label=r"$\sqrt{2/\nu}$")
    ax.set_xlabel(r"effective interferers $\nu$")
    ax.set_ylabel(r"rel.std[$\xi_{\rm block}$]")
    ax.set_title(r"$\nu=(\sum\kappa)^2/\sum\kappa^2$ predicts the spread")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.semilogx([r["ratio"] for r in rate_rows],
                [r["bit_per_s_bound"] for r in rate_rows], "o-",
                color="#1e8449", label="Laurent–Massart (theorem)")
    ax.semilogx([r["ratio"] for r in rate_rows],
                [r["bit_per_s_gamma"] for r in rate_rows], "s--",
                color="#b9770e", label="Gamma moment match")
    ax.set_xlabel("power ratio (1 = equal)")
    ax.set_ylabel("per-user rate (bit/s)")
    ax.set_title("25 km: rate vs interferer imbalance")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot([r["nu"] for r in rate_rows],
            [100 * r["fraction_of_equal_bound"] for r in rate_rows], "o-",
            color="#1e8449")
    ax.axhline(100, color="grey", ls=":", lw=1)
    ax.set_xlabel(r"effective interferers $\nu$")
    ax.set_ylabel("% of equal-power rate")
    ax.set_title("imbalance is not a small correction")

    plotting.save(fig, "e13_kappa_concentration")

    out = dict(distance_km=distance_km, n_symbols=n_symbols, L=L,
               n_users=int(cdma.n_users), n_interferers=K, eps_mui=EPS_MUI,
               eps_ref=EPS_REF, mc_draws=MC_DRAWS,
               calibration=calib_rows, mc_corroboration=mc_rows,
               identity=identity_rows, rate=rate_rows,
               ratio_star=ratio_star, nu_star=nu_star,
               false_security=false_rows,
               xi_design_equal_power=xi_design,
               runtime_s=time.perf_counter() - t0)
    print(f"  runtime {out['runtime_s']:.2f}s")
    return out


if __name__ == "__main__":
    dump("e13_kappa_concentration", main())
