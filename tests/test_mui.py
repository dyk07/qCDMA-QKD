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
