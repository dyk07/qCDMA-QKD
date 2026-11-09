"""Validation tests for the MUI statistics.

The analytic moments are checked against Monte Carlo, and the block-averaged
xi model -- the object that actually enters the key rate -- is checked against
its closed form.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from qcdma_qkd import mui

L = 1024
V_A = 4.0
SIG2 = mui.quadrature_variance(V_A)


def test_variance_scales_as_inverse_L():
    kappa = mui.uniform_kappa(7, 1.0)
    v1 = mui.mui_variance(kappa, 1024, SIG2)
    v2 = mui.mui_variance(kappa, 2048, SIG2)
    assert v1 == pytest.approx(2.0 * v2, rel=1e-12)


def test_xi_independent_of_symbol_variance():
    """xi_MUI = (1/L) sum kappa must not depend on sigma."""
    kappa = mui.uniform_kappa(7, 1.0)
    assert mui.xi_from_codes(kappa, L) == pytest.approx(7.0 / L)


def test_kurtosis_formula():
    for K in (1, 2, 5, 16):
        kappa = mui.uniform_kappa(K, 1.0)
        assert mui.mui_kurtosis(kappa, L, SIG2) == pytest.approx(6.0 / K)


@pytest.mark.parametrize("K", [1, 3, 7])
def test_moments_match_monte_carlo(K):
    kappa = mui.uniform_kappa(K, 1.0)
    x = mui.monte_carlo_mui(kappa, L, SIG2, n_draws=400_000, seed=K)
    assert np.var(x) == pytest.approx(mui.mui_variance(kappa, L, SIG2), rel=0.02)
    assert stats.kurtosis(x) == pytest.approx(mui.mui_kurtosis(kappa, L, SIG2), abs=0.1)


def test_single_interferer_tail_matches_bessel_closed_form():
    r"""For K=1 the tail is exactly (2/pi) * integral_t^inf K_0."""
    kappa = mui.uniform_kappa(1, 1.0)
    v = mui.mui_variance(kappa, L, SIG2)
    for tau in (0.01, 0.03, 0.08):
        a = mui.tail_mixture(tau, kappa, L, SIG2)
        b = mui.tail_single_interferer_exact(tau / np.sqrt(v), v)
        assert a == pytest.approx(b, rel=0.02)


@pytest.mark.parametrize("tau", [0.1, 0.2, 0.3])
def test_mixture_tail_matches_monte_carlo(tau):
    kappa = mui.uniform_kappa(7, 1.0)
    x = mui.monte_carlo_mui(kappa, L, SIG2, n_draws=400_000, seed=11)
    assert mui.tail_mixture(tau, kappa, L, SIG2) == pytest.approx(
        mui.empirical_outage(x, tau), rel=0.08)


def test_tail_bounds_are_valid_upper_bounds():
    """Markov bounds must never under-estimate the true tail."""
    kappa = mui.uniform_kappa(7, 1.0)
    x = mui.monte_carlo_mui(kappa, L, SIG2, n_draws=200_000, seed=5)
    for tau in (0.15, 0.25, 0.35):
        true = mui.empirical_outage(x, tau)
        assert mui.tail_fourth_moment(tau, kappa, L, SIG2) >= true - 0.02
        assert mui.tail_chebyshev(tau, kappa, L, SIG2) >= true - 0.02


def test_block_averaging_does_not_reduce_uncertainty():
    """The central claim: more symbols do NOT shrink xi_block's spread."""
    kappa = mui.uniform_kappa(7, 1.0)
    a = mui.xi_block_samples(kappa, L, n_draws=50_000, seed=1).std()
    b = mui.xi_block_samples(kappa, L, n_draws=50_000, seed=2).std()
    assert a == pytest.approx(b, rel=0.05)


def test_xi_block_relative_std_matches_sqrt_2_over_K():
    K = 7
    kappa = mui.uniform_kappa(K, 1.0)
    rel = mui.xi_block_std(kappa, L) / mui.xi_block_mean(kappa, L)
    assert rel == pytest.approx(np.sqrt(2.0 / K), rel=1e-9)


def test_required_xi_ordering():
    """The required (tail) xi must exceed the mean, and decrease with eps."""
    kappa = mui.uniform_kappa(7, 1.0)
    reqs = [mui.required_xi(e, kappa, L) for e in (1e-3, 1e-6, 1e-9, 1e-12)]
    assert all(r > mui.xi_block_mean(kappa, L) for r in reqs)
    assert all(a < b for a, b in zip(reqs, reqs[1:]))
    assert mui.required_xi(1e-12, kappa, L) < 1.0


def test_more_chips_suppress_mui():
    kappa = mui.uniform_kappa(7, 1.0)
    assert mui.xi_block_mean(kappa, 4096) == pytest.approx(
        mui.xi_block_mean(kappa, 1024) / 4.0)


# --------------------------------------------------------------------------- #
# the rigorous bound must actually dominate the exact quantile
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("K", [2, 7, 15])
@pytest.mark.parametrize("eps", [1e-6, 1e-9, 1e-12, 1e-15])
def test_laurent_massart_bound_dominates_the_exact_quantile(K, eps):
    """The bound is a theorem and must never fall below the truth.

    Regression: writing the two log terms with coefficient 1 instead of 2 gives
    0.0435 vs the exact 0.0540 at K=2, eps=1e-12 -- i.e. a "bound" that is
    violated.  This test is the guard against mis-copying the constants.
    """
    kap = mui.uniform_kappa(K, 1.0)
    L = 1024
    bound = mui.xi_block_bound(eps, kap, L)
    exact = mui.xi_block_exact(eps, kap, L)
    assert bound >= exact, (
        f"bound {bound:.5f} < exact {exact:.5f} at K={K}, eps={eps:g}")


@pytest.mark.parametrize("eps", [1e-6, 1e-12])
def test_gaussian_quantile_is_documented_as_unsafe(eps):
    """The Gaussian fit is a model and errs optimistically in the tail.

    If this ever starts passing as "safe", the docs' warning is wrong and must
    be revised -- so the assertion is that it is *not* safe, at these settings.
    """
    kap = mui.uniform_kappa(7, 1.0)
    d = mui.xi_gap_vs_bound(eps, kap, 1024)
    assert d["exact_is_closed_form"]
    assert not d["gaussian_is_safe"]
    assert d["gaussian_gap"] < -0.2          # understates by more than 20%
    assert d["bound_is_safe"]


def test_exact_quantile_matches_chi2_for_equal_power():
    """Equal kappa -> xi_block * L / kappa is exactly chi^2_K."""
    from scipy import stats
    K, L, eps = 7, 1024, 1e-12
    want = stats.chi2.ppf(1 - eps, df=K) / L
    assert mui.xi_block_exact(eps, mui.uniform_kappa(K, 1.0), L) == pytest.approx(want)


def test_bound_holds_for_unequal_power_too():
    """The Gamma route is only approximate when kappa differs; the bound is not."""
    kap = np.array([1.0, 0.5, 0.2, 1.5, 0.9, 0.3, 0.7])
    d = mui.xi_gap_vs_bound(1e-12, kap, 1024)
    assert not d["exact_is_closed_form"]
    assert d["bound_is_safe"]
