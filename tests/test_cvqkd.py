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
