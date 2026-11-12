"""Interferer power imbalance: ``nu`` controls the MUI fluctuation.

Why this file exists
--------------------
Every headline number in the project assumes equal interferer powers
(``kappa_j = 1``).  For equal powers ``xi_block`` is exactly
``(kappa/L) * chi^2_K`` and the ``eps``-quantile is a closed form; for unequal
powers ``mui.xi_block_exact`` falls back to a **moment-matched Gamma**, which is
an approximation and is *optimistic* in the tail.  Nothing pinned either fact,
so the equal-power assumption could quietly become load-bearing without the
suite noticing.

The organising quantity is the effective interferer count

    nu = (sum_j kappa_j)^2 / sum_j kappa_j^2,      1 <= nu <= K

with the exact identity

    rel.std[xi_block] = sqrt(2 / nu).

``nu = K`` is equal powers, ``nu -> 1`` is one dominant interferer.

These tests pin:
* the identity (so ``nu`` is the right summary statistic),
* the direction and size of the Gamma approximation error,
* that Laurent--Massart is always above the true quantile,
* that concentration makes the design optimistic (never the reverse).

On the choice of reference
--------------------------
The first draft of these tests used Monte Carlo at ``eps = 1e-6``.  That is not
sound: two million draws contain about **two** tail events at that ``eps``, so
``np.quantile`` returns the largest order statistic, which fluctuated by ~13%
between seeds and once produced a non-monotone error curve that looked like a
physical effect.  The reference here is now ``mui.xi_block_quantile_saddle``,
which reproduces exact ``chi^2_K / L`` to 0.02% and agrees with Monte Carlo to
2% where Monte Carlo *is* trustworthy (``eps = 1e-4``).
"""

from __future__ import annotations

import numpy as np
import pytest

from qcdma_qkd import mui

L = 1024
K = 7
EPS = 1e-6             # comparison point for the three quantile routes
EPS_MC = 1e-4          # the smallest eps Monte Carlo resolves with 2e6 draws


def geometric_kappa(ratio: float, n_interferers: int = K) -> np.ndarray:
    """Geometric decay, normalised to the same total power as equal kappa."""
    w = np.power(float(ratio), np.arange(n_interferers))
    return n_interferers * w / w.sum()


def effective_nu(kappa: np.ndarray) -> float:
    return float(np.sum(kappa) ** 2 / np.sum(kappa ** 2))


# --------------------------------------------------------------------------- #
# 1. nu is the right summary statistic
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ratio", [1.0, 0.8, 0.5, 0.3, 0.1, 1e-3])
def test_relative_std_equals_sqrt_two_over_nu(ratio):
    """``rel.std = sqrt(2/nu)`` exactly -- this is why ``nu`` is enough."""
    kappa = geometric_kappa(ratio)
    rel_std = mui.xi_block_std(kappa, L) / mui.xi_block_mean(kappa, L)
    assert rel_std == pytest.approx(np.sqrt(2.0 / effective_nu(kappa)), rel=1e-12)


@pytest.mark.parametrize("ratio", [1.0, 0.5, 0.1])
def test_nu_bounds_one_and_k(ratio):
    nu = effective_nu(geometric_kappa(ratio))
    assert 1.0 - 1e-12 <= nu <= K + 1e-12


def test_equal_power_gives_nu_equal_to_k():
    assert effective_nu(geometric_kappa(1.0)) == pytest.approx(K, rel=1e-12)


def test_full_concentration_approaches_one_interferer():
    """As one interferer dominates, ``nu -> 1`` and rel.std -> sqrt(2) = 141%."""
    kappa = geometric_kappa(1e-6)
    assert effective_nu(kappa) == pytest.approx(1.0, abs=1e-3)
    rel_std = mui.xi_block_std(kappa, L) / mui.xi_block_mean(kappa, L)
    assert rel_std == pytest.approx(np.sqrt(2.0), abs=2e-3)


def test_concentration_always_increases_the_fluctuation():
    """More imbalance = fewer effective interferers = noisier. Never reversed."""
    stds = [mui.xi_block_std(geometric_kappa(r), L)
            / mui.xi_block_mean(geometric_kappa(r), L)
            for r in (1.0, 0.8, 0.5, 0.3, 0.1)]
    assert all(a < b for a, b in zip(stds, stds[1:]))


def test_mean_is_invariant_under_rebalancing():
    """The generator keeps ``sum(kappa)`` fixed, so the mean MUI cannot move.

    If this ever fails, E13's rate changes would mix two effects (more power
    *and* more imbalance) and the whole comparison would be meaningless.
    """
    means = [mui.xi_block_mean(geometric_kappa(r), L)
             for r in (1.0, 0.5, 0.1, 1e-3)]
    assert all(m == pytest.approx(means[0], rel=1e-12) for m in means)


# --------------------------------------------------------------------------- #
# 2. The three quantile routes: which one is safe?
# --------------------------------------------------------------------------- #
def test_saddlepoint_reference_reproduces_the_exact_chi_square():
    """The reference must be near-exact where the answer is known.

    Equal powers give ``chi^2_K / L`` exactly.  If the reference disagreed here
    it could not serve as a baseline for the unequal-power cases.
    """
    kappa = geometric_kappa(1.0)
    for eps in (1e-4, 1e-6, 1e-9, 1e-12):
        exact = mui.xi_block_exact(eps, kappa, L)
        ref = mui.xi_block_quantile_saddle(eps, kappa, L)
        assert ref == pytest.approx(exact, rel=2e-3), f"eps={eps}"


def test_saddlepoint_reference_agrees_with_monte_carlo_where_mc_works():
    """Corroborate the reference independently at a resolvable quantile."""
    rng = np.random.default_rng(0)
    for ratio in (1.0, 0.8, 0.5, 0.3):
        kappa = geometric_kappa(ratio)
        z = rng.normal(size=(2_000_000, K))
        mc = np.quantile((kappa * z ** 2).sum(axis=1) / L, 1.0 - EPS_MC)
        ref = mui.xi_block_quantile_saddle(EPS_MC, kappa, L)
        assert ref == pytest.approx(mc, rel=0.05), f"ratio={ratio}"


def test_laurent_massart_is_above_the_true_quantile():
    """The bound must be valid for every kappa, not just equal ones."""
    for ratio in (1.0, 0.9, 0.8, 0.5, 0.3, 0.1):
        kappa = geometric_kappa(ratio)
        for eps in (1e-6, 1e-12):
            ref = mui.xi_block_quantile_saddle(eps, kappa, L)
            assert mui.xi_block_bound(eps, kappa, L) >= ref, f"ratio={ratio}"


def test_gamma_route_understates_the_quantile_for_unequal_power():
    """The moment-matched Gamma is *optimistic* once the powers differ.

    This is the important direction: an optimistic MUI quantile means the
    design provisions too little excess noise and overstates the key rate.
    Measured against the saddlepoint reference, not against Monte Carlo.

    The error is not monotone in the imbalance -- it peaks near ``ratio = 0.5``
    at about -19% and *shrinks* as one interferer becomes fully dominant,
    because ``nu -> 1`` makes the Gamma fit a single chi-square again.
    """
    errors = {}
    for ratio in (0.8, 0.5, 0.3, 0.1):
        kappa = geometric_kappa(ratio)
        gam = mui.xi_block_exact(EPS, kappa, L)
        ref = mui.xi_block_quantile_saddle(EPS, kappa, L)
        errors[ratio] = gam / ref - 1.0
        assert errors[ratio] < 0.0, f"Gamma was not optimistic at ratio={ratio}"
    # The worst case is interior, not at the extreme -- pin that shape.
    assert min(errors, key=errors.get) == 0.5
    assert errors[0.5] < -0.15


def test_gamma_route_is_exact_for_equal_power():
    """For equal powers the quantile *is* chi^2_K/L, so no approximation."""
    kappa = geometric_kappa(1.0)
    from scipy import stats
    for eps in (1e-6, 1e-12):
        exact = stats.chi2.ppf(1.0 - eps, df=K) / L
        assert mui.xi_block_exact(eps, kappa, L) == pytest.approx(exact, rel=1e-12)


def test_required_xi_still_grows_with_concentration():
    """Whatever the route, less balance must mean more required excess noise."""
    xis = [mui.xi_block_bound(1e-12, geometric_kappa(r), L)
           for r in (1.0, 0.8, 0.5, 0.3, 0.1)]
    assert all(a < b for a, b in zip(xis, xis[1:]))


# --------------------------------------------------------------------------- #
# 3. The design must be optimistic, never pessimistic
# --------------------------------------------------------------------------- #
def test_equal_power_design_underestimates_an_imbalanced_link():
    """A link designed for equal powers is NOT safe under imbalance.

    At ``eps_MUI = 1e-12`` the equal-power design provisions
    ``xi = 0.0692``; with ratio 0.8 the true requirement is already ``0.1321``.
    This is the false-security gap the docs quantify -- it must stay positive.
    """
    xi_design = mui.xi_block_exact(1e-12, geometric_kappa(1.0), L)
    for ratio in (0.9, 0.8, 0.5, 0.3):
        required = mui.xi_block_bound(1e-12, geometric_kappa(ratio), L)
        assert required > xi_design, f"no gap at ratio={ratio}"
    # And the same must hold against the *reference*, not only the bound: even
    # an optimistically computed requirement already exceeds the design.
    for ratio in (0.8, 0.5, 0.3):
        required = mui.xi_block_quantile_saddle(1e-12, geometric_kappa(ratio), L)
        assert required > xi_design, f"reference shows no gap at ratio={ratio}"


def test_false_security_gap_grows_with_concentration():
    xi_design = mui.xi_block_exact(1e-12, geometric_kappa(1.0), L)
    gaps = [mui.xi_block_bound(1e-12, geometric_kappa(r), L) / xi_design - 1.0
            for r in (0.8, 0.5, 0.3)]
    assert all(a < b for a, b in zip(gaps, gaps[1:]))
