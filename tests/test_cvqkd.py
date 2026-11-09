"""Validation tests for the CV-QKD rate model.

Every assertion here is a *physics* check against a closed-form reference, so
these tests also serve as the documentation of the modelling conventions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qcdma_qkd import cvqkd
from qcdma_qkd.config import CVQKDConfig


def test_ideal_limit_matches_pure_loss_gg02():
    """V_A=4, T=1, xi=0, eta=1, v_el=0, beta=1 must give 0.5*log2(5)."""
    r = cvqkd.asymptotic_rate(V_A=4.0, T=1.0, xi=0.0, eta=1.0, v_el=0.0, beta=1.0)
    assert r["S"] == pytest.approx(0.5 * math.log2(5.0), rel=1e-9)


def test_ideal_limit_other_modulation():
    """Same check for an arbitrary modulation variance."""
    for V_A in (1.0, 2.0, 8.0, 20.0):
        r = cvqkd.asymptotic_rate(V_A=V_A, T=1.0, xi=0.0, eta=1.0, v_el=0.0,
                                  beta=1.0)
        assert r["S"] == pytest.approx(0.5 * math.log2(1.0 + V_A), rel=1e-9)


def test_rate_monotone_decreasing_in_noise():
    """More excess noise can never help."""
    cfg = CVQKDConfig()
    T = cvqkd.total_transmittance(20.0, cfg.alpha_db_per_km, cfg.eta)
    rates = [cvqkd.asymptotic_rate(cfg.V_A, T, xi, cfg.eta, cfg.v_el, cfg.beta)["S_pos"]
             for xi in (0.0, 0.01, 0.05, 0.1, 0.2)]
    assert all(a >= b - 1e-12 for a, b in zip(rates, rates[1:]))


def test_rate_monotone_decreasing_in_distance():
    cfg = CVQKDConfig()
    d = np.array([0.0, 5.0, 10.0, 25.0, 50.0, 100.0])
    r = cvqkd.rate_vs_distance(cfg, d)
    assert all(a >= b - 1e-12 for a, b in zip(r, r[1:]))


def test_holevo_non_negative_and_below_iab():
    """chi_BE >= 0 and the raw rate cannot exceed I_AB."""
    cfg = CVQKDConfig()
    for L in (0.0, 10.0, 40.0, 80.0):
        r = cvqkd.asymptotic_rate(cfg.V_A,
                                  cvqkd.total_transmittance(L, cfg.alpha_db_per_km, cfg.eta),
                                  cfg.excess_noise, cfg.eta, cfg.v_el, cfg.beta)
        assert r["chi_BE"] >= -1e-12
        assert r["S"] <= r["I_AB"] + 1e-12


def test_entropy_function_limits():
    assert cvqkd.g(0.0) == pytest.approx(0.0)
    big = 1e6
    # g(x) -> log2(x) + 1/log(2) for large x
    assert cvqkd.g(big) == pytest.approx(math.log2(big) + 1.0 / math.log(2.0), rel=1e-4)


def test_parameter_estimation_widens_with_fewer_samples():
    a = cvqkd.pe_confidence_intervals(0.5, 0.01, n_pe=10 ** 6)
    b = cvqkd.pe_confidence_intervals(0.5, 0.01, n_pe=10 ** 4)
    assert b["rel_err"] > a["rel_err"]
    assert b["xi_ub"] >= a["xi_ub"]
    assert b["T_lb"] <= a["T_lb"]


def test_finite_size_below_asymptotic():
    cfg = CVQKDConfig()
    T = cvqkd.total_transmittance(10.0, cfg.alpha_db_per_km, cfg.eta)
    a = cvqkd.asymptotic_rate(cfg.V_A, T, cfg.excess_noise, cfg.eta, cfg.v_el, cfg.beta)
    f = cvqkd.finite_size_rate(cfg.V_A, T, cfg.excess_noise, cfg.eta, cfg.v_el,
                               cfg.beta, n_symbols=10 ** 7, eps_sm=1e-10,
                               eps_pa=1e-12)
    assert f["S"] < a["S"]


def test_heterodyne_is_refused():
    with pytest.raises(NotImplementedError):
        cvqkd.holevo_bound(4.0, 0.5, 0.01, 0.9, 0.05, detection="heterodyne")


def test_transmittance_conventions():
    """total_transmittance must include eta exactly once."""
    eta = 0.9
    assert cvqkd.total_transmittance(0.0, 0.2, eta) == pytest.approx(eta)
    assert cvqkd.db_to_transmittance(0.0, 0.2) == pytest.approx(1.0)
    # 10 km at 0.2 dB/km = 2 dB -> 10**(-2/10) = 10**(-0.2)
    assert cvqkd.total_transmittance(10.0, 0.2, eta) == pytest.approx(
        eta * 10 ** (-0.2))


# --------------------------------------------------------------------------- #
# PE bounds: the exact chi-square route must dominate the Gaussian one
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_pe", [10 ** 4, 10 ** 6, 10 ** 7])
def test_exact_pe_noise_bound_dominates_the_gaussian_one(n_pe):
    """A sample variance is chi-square, not Gaussian.

    Regression: the Gaussian interval is *narrower* on the xi coordinate, i.e.
    it understates the excess noise and therefore overstates the key rate.
    This test pins that direction so the docs warning cannot rot.
    """
    cfg = CVQKDConfig()
    T = cvqkd.total_transmittance(25.0, cfg.alpha_db_per_km, cfg.eta)
    d = cvqkd.pe_gap_vs_exact(T, 0.0692, n_pe, 1e-10, cfg.eta, cfg.V_A, cfg.v_el)
    assert d["xi_ub_exact"] >= d["xi_ub_gaussian"], (
        "the Gaussian PE bound is supposed to be the anti-conservative one")
    assert not d["xi_gaussian_is_safe"]
    assert d["modulus"] >= 1.0


def test_exact_pe_bounds_tighten_with_more_samples():
    """More PE samples must narrow both bounds monotonically."""
    cfg = CVQKDConfig()
    T = cvqkd.total_transmittance(25.0, cfg.alpha_db_per_km, cfg.eta)
    prev = None
    for n_pe in (10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7):
        e = cvqkd.pe_confidence_intervals_exact(T, 0.0692, n_pe, 1e-10,
                                                cfg.eta, cfg.V_A, cfg.v_el)
        if prev is not None:
            assert e["xi_ub"] <= prev["xi_ub"] + 1e-12
            assert e["T_lb"] >= prev["T_lb"] - 1e-12
        prev = e


def test_finite_size_rate_with_pe_bounds_is_below_nominal():
    """Feeding the confidence bounds must not increase the rate.

    The nominal-parameter rate is an upper bound on the achievable rate: the
    real protocol does not know T and xi, only intervals for them.
    """
    cfg = CVQKDConfig()
    T = cvqkd.total_transmittance(25.0, cfg.alpha_db_per_km, cfg.eta)
    a = cvqkd.finite_size_rate(cfg.V_A, T, 0.0692, cfg.eta, cfg.v_el, cfg.beta,
                         cfg.detection, n_symbols=10 ** 8, eps_sm=1e-10,
                         eps_pa=1e-12, sample_fraction=0.1)
    b = cvqkd.finite_size_rate(cfg.V_A, T, 0.0692, cfg.eta, cfg.v_el, cfg.beta,
                         cfg.detection, n_symbols=10 ** 8, eps_sm=1e-10,
                         eps_pa=1e-12, sample_fraction=0.1, eps_pe=1e-10)
    assert b["S_pos"] <= a["S_pos"]
    assert b["pe_applied"] and not a["pe_applied"]
    assert b["T_used"] <= a["T_used"]
    assert b["xi_used"] >= a["xi_used"]
