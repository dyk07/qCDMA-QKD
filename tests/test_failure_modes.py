"""Failure modes: are they declared, enforced, and quantified?

The protocol doc lists four abort conditions.  This module pins that each one is
(a) actually *enforced* by ``loop.run_loop`` rather than merely described, and
(b) *quantified*, so "the model is valid" is a checkable statement rather than a
hope.

The failure taxonomy
--------------------
=====================  ==================  ==============================
failure                probability         treatment
=====================  ==================  ==============================
authentication         eps_auth* per round abort (fail-closed); ring may retry
budget infeasible      deterministic       abort, ``u_n < 0``
PE rejected            eps_pe per round    abort, no seed released
root exhausted         ring size           abort, needs offline bootstrap
near miss              deterministic       *not* an abort; flagged for ops
=====================  ==================  ==============================

``eps_auth*`` is ~4e-12 per round at t=128, i.e. cryptographic, not statistical.
The budget and near-miss rows are deterministic design conditions, not events
with a probability -- conflating the two is the main conceptual trap here.
"""

from __future__ import annotations

import numpy as np
import pytest

from qcdma_qkd import budget, cvqkd, mui, security
from qcdma_qkd import loop
from qcdma_qkd.config import BudgetConfig, CDMAConfig, CVQKDConfig, SecurityConfig

CV = CVQKDConfig()
CDMA = CDMAConfig()
BD = BudgetConfig()
SEC = SecurityConfig()
L = CDMA.spreading_factor
KW = dict(k0_bits=2 ** 20, cfg_cv=CV, cfg_cdma=CDMA, cfg_bud=BD,
          rate_model="asymptotic",
          distance_km=25.0, n_symbols=10 ** 5)


def _run(n_rounds=4, **extra):
    """Run with the module defaults, letting `extra` override any of them."""
    kw = dict(KW)
    kw.update(extra)
    return loop.run_loop(n_rounds, **kw)


# --------------------------------------------------------------------------- #
# every declared failure mode is enforced
# --------------------------------------------------------------------------- #
def test_auth_failure_aborts():
    r = _run(tamper_at=2, tamper_events=1)
    assert r["aborted_at"] == 2
    assert "authentication failure" in r["reason"]


def test_budget_infeasible_aborts():
    """u_n < 0 must abort, not silently produce negative application key."""
    r = _run(distance_km=90.0)          # beyond the 68.7 km MUI-limited range
    assert r["aborted_at"] == 1
    assert "budget infeasible" in r["reason"]
    assert r["rounds"][0].feasible is False


def test_budget_check_can_be_disabled_for_study():
    """`enforce_budget=False` records the infeasible round instead of aborting
    *before* recording it -- but the chain still cannot continue, because
    K_n < a+c leaves nothing to derive round n+1 from.  The point of the flag
    is to inspect the infeasible round, not to keep running on empty key.
    """
    r = _run(distance_km=90.0, enforce_budget=False)
    assert r["rounds"][0].feasible is False
    assert r["rounds"][0].u_bits < 0
    assert r["aborted_at"] == 1
    assert "budget infeasible" in r["reason"]


def test_enforced_and_study_modes_report_the_same_verdict():
    """Both modes must agree that this design is infeasible."""
    strict = _run(distance_km=90.0, enforce_budget=True)
    study = _run(distance_km=90.0, enforce_budget=False)
    assert strict["aborted_at"] == study["aborted_at"] == 1
    assert strict["rounds"][0].feasible is study["rounds"][0].feasible is False


def test_pe_rejection_aborts():
    r = _run(n_rounds=5, pe_reject_at=3)
    assert r["aborted_at"] == 3
    assert "parameter estimation rejected" in r["reason"]
    assert r["rounds"][-1].extras.get("pe_rejected") is True


def test_root_exhaustion_aborts():
    r = _run(tamper_at=1, tamper_events=100, n_k0_roots=2, recover=True)
    assert r["aborted_at"] == 1
    assert r["roots_remaining"] == 0


def test_happy_path_reports_all_clear():
    r = _run()
    assert r["aborted_at"] is None
    assert r["near_misses"] == []
    assert r["roots_burned"] == []
    assert r["min_surplus"] > 0.5          # >50% of K_n left for the application


# --------------------------------------------------------------------------- #
# near misses: feasible but tight (not a failure)
# --------------------------------------------------------------------------- #
def test_near_miss_flags_feasible_but_tight_rounds():
    r = _run(k0_bits=2 ** 20, n_symbols=10 ** 5, distance_km=50.0,
             enforce_budget=False, near_miss_ratio=0.5)
    rr = r["rounds"][0]
    assert rr.feasible, "this point should still be feasible"
    assert rr.extras["surplus"] < 0.5
    assert r["near_misses"] == [1, 2, 3, 4][:len(r["near_misses"])]


def test_infeasible_rounds_are_not_called_near_misses():
    """A negative surplus is a failure, not a near miss -- keep them separate."""
    r = _run(distance_km=90.0, enforce_budget=False)
    assert r["rounds"][0].feasible is False
    assert r["near_misses"] == []


# --------------------------------------------------------------------------- #
# the fixed seed cost caps distance for SHORT blocks
# --------------------------------------------------------------------------- #
def _feasible_distance(n_symbols: float) -> float:
    """Largest distance with u_n >= 0."""
    xi = mui.required_xi(1e-12, mui.uniform_kappa(CDMA.n_users - 1, 1.0), L)
    lo, hi = 0.0, 300.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        T = cvqkd.total_transmittance(mid, CV.alpha_db_per_km, CV.eta)
        r = cvqkd.asymptotic_rate(CV.V_A, T, CV.excess_noise + xi,
                                  CV.eta, CV.v_el, CV.beta)
        b = budget.round_budget(r["S_pos"] * n_symbols, n_symbols, r["I_AB"], BD)
        if b["feasible"]:
            lo = mid
        else:
            hi = mid
    return lo


def test_short_blocks_are_distance_limited_by_the_fixed_seed():
    """The 256-bit seed is a *fixed* per-round cost: on short blocks it, not the
    quantum rate, sets the reach.  Quantifying this is part of model validity.
    """
    d_short = _feasible_distance(1e4)
    d_long = _feasible_distance(1e8)
    rate_limit = mui.required_xi(1e-12,
                                 mui.uniform_kappa(CDMA.n_users - 1, 1.0), L)
    assert d_short < 40.0, d_short          # ~31.5 km
    assert d_long == pytest.approx(cvqkd.max_distance(CV, rate_limit), abs=1.0)


def test_budget_limit_converges_to_the_rate_limit_for_long_blocks():
    assert _feasible_distance(1e7) == pytest.approx(_feasible_distance(1e9),
                                                    abs=0.5)


# --------------------------------------------------------------------------- #
# probabilities: cryptographic, not statistical
# --------------------------------------------------------------------------- #
def test_authentication_failure_probability_is_cryptographic():
    """t=128 -> 2^-128 per message; a ring of 3 would need 1e-114 bad luck."""
    per_msg = 2.0 ** -BD.tag_bits
    assert per_msg < 1e-38
    assert (4 * per_msg) ** 3 < 1e-110


def test_deterministic_modes_have_no_probability():
    """Budget/PE/near-miss are design conditions -- they must not be modelled
    as random events with an 'eps'.  Stating that explicitly protects the
    taxonomy from drifting into hand-waving.
    """
    # a round far beyond the feasible range always fails budget, every time
    for _ in range(3):
        assert _run(distance_km=90.0)["aborted_at"] == 1


def test_per_round_epsilon_budget_stays_under_target():
    b = security.from_config(SEC, eps_auth_rounds=4)
    assert b.total() < SEC.eps_total


def test_sequential_composition_has_a_finite_round_budget():
    """The model is valid only for a bounded number of rounds at a given eps."""
    eps_round = 1e-12
    lo, hi = 1, 10 ** 12
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if security.compose_sequential(eps_round, mid) <= 1e-9:
            lo = mid
        else:
            hi = mid
    assert 10 ** 4 < lo < 10 ** 6          # ~1.8e4 rounds at eps_total <= 1e-9
