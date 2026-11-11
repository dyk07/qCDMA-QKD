"""The closed loop must actually use its own security parameters.

Why this file exists
--------------------
``run_loop`` accepted ``eps_pe`` for a long time and **never passed it on**:
``simulate_round`` did not even have a matching argument.  The loop therefore
reported the nominal-parameter rate (``S = 0.033316``) while every document
quoted ``eps_pe = 1e-10`` in the security statement -- a target the protocol
was not provisioning for.  The gap is 38% of the key rate, far larger than the
finite-size correction that *was* being tracked.

Nothing caught it because the code ran, the numbers were self-consistent, and
the tests only asserted that the loop completes.  These tests assert the
*linkage* instead: the value the caller supplies must reach the rate function
and change ``T_used`` / ``xi_used``.

The same class of defect applies to ``xi_mui_model``, so it is covered here
too.  A parameter that is accepted but ignored is worse than a missing one:
it makes the signature lie.
"""

from __future__ import annotations

import numpy as np
import pytest

from qcdma_qkd import cvqkd, loop, mui
from qcdma_qkd.config import BudgetConfig, CDMAConfig, CVQKDConfig, SecurityConfig

L = 1024
N_SYMBOLS = 10 ** 8
DISTANCE = 25.0


def run(**kw):
    return loop.run_loop(3, 2 ** 20, CVQKDConfig(), CDMAConfig(),
                         BudgetConfig(), DISTANCE, n_symbols=N_SYMBOLS, **kw)


# --------------------------------------------------------------------------- #
# 1. PE confidence bounds reach the rate
# --------------------------------------------------------------------------- #
def test_pe_bounds_are_applied_by_default():
    """The default round must report ``pe_applied`` -- not merely accept the arg."""
    r = run()
    assert r["rounds"], "no rounds produced"
    for rd in r["rounds"]:
        assert rd.extras["pe_applied"] is True
        assert rd.extras["eps_pe"] == pytest.approx(1e-10)


def test_pe_bounds_actually_moved_the_parameters():
    """``T_used``/``xi_used`` must differ from the nominal values.

    This is the assertion that would have caught the original bug: it is
    possible to set a flag without changing the computation.
    """
    rd = run()["rounds"][0]
    T_nominal = rd.extras["T"]
    assert rd.extras["T_used"] < T_nominal          # lower transmittance
    assert rd.extras["xi_used"] > rd.xi_total       # higher excess noise


def test_pe_bounds_reduce_the_key_rate():
    """A confidence bound is conservative, so it can only lower ``S``."""
    strict = run()["rounds"][0]
    nominal = run(eps_pe=None)["rounds"][0]
    assert nominal.extras["pe_applied"] is False
    assert strict.rate_per_symbol < nominal.rate_per_symbol
    # and the reduction is substantial, not a rounding effect
    assert strict.rate_per_symbol / nominal.rate_per_symbol == pytest.approx(
        0.7236, abs=5e-3)


def test_pe_penalty_matches_the_standalone_function():
    """The loop and a direct ``finite_size_rate`` call must agree exactly.

    Guards against the loop re-deriving any part of the calculation on its own.
    """
    cfg, cdma, bud = CVQKDConfig(), CDMAConfig(), BudgetConfig()
    sec = SecurityConfig()
    kappa = mui.uniform_kappa(cdma.n_users - 1, 1.0)
    xi_tot = cfg.excess_noise + mui.required_xi(1e-12, kappa, L)
    T = cvqkd.total_transmittance(DISTANCE, cfg.alpha_db_per_km, cfg.eta)
    direct = cvqkd.finite_size_rate(
        cfg.V_A, T, xi_tot, cfg.eta, cfg.v_el, cfg.beta, cfg.detection,
        n_symbols=N_SYMBOLS, eps_sm=sec.eps_sm, eps_pa=sec.eps_pa,
        sample_fraction=bud.pe_sample_fraction, eps_pe=sec.eps_qkd)
    rd = run()["rounds"][0]
    assert rd.rate_per_symbol == pytest.approx(direct["S_pos"], rel=1e-12)
    assert rd.k_bits == pytest.approx(direct["S_pos"] * N_SYMBOLS, rel=1e-12)


def test_tightening_eps_pe_only_reduces_the_rate():
    """Smaller eps_pe = wider interval = lower rate.  Monotone, never reversed."""
    rates = [run(eps_pe=e)["rounds"][0].rate_per_symbol
             for e in (1e-6, 1e-9, 1e-12)]
    assert all(a > b for a, b in zip(rates, rates[1:]))


def test_eps_pe_none_reproduces_the_old_nominal_behaviour():
    rd = run(eps_pe=None)["rounds"][0]
    assert rd.extras["T_used"] == pytest.approx(rd.extras["T"])
    assert rd.extras["xi_used"] == pytest.approx(rd.xi_total)


# --------------------------------------------------------------------------- #
# 2. The MUI quantile route reaches the rate
# --------------------------------------------------------------------------- #
def test_xi_mui_model_selects_the_right_quantile():
    kappa = mui.uniform_kappa(CDMAConfig().n_users - 1, 1.0)
    exact = run(xi_mui_model="quantile")["rounds"][0].xi_mui
    bound = run(xi_mui_model="bound")["rounds"][0].xi_mui
    assert exact == pytest.approx(mui.required_xi(1e-12, kappa, L))
    assert bound == pytest.approx(mui.xi_block_bound(1e-12, kappa, L))
    # the theorem is looser, so the conservative route is slower
    assert bound > exact


def test_bound_route_is_slower_than_the_quantile_route():
    q = run(xi_mui_model="quantile")["rounds"][0].rate_per_symbol
    b = run(xi_mui_model="bound")["rounds"][0].rate_per_symbol
    assert b < q


def test_unknown_xi_mui_model_is_rejected():
    with pytest.raises(ValueError, match="xi_mui_model must be"):
        run(xi_mui_model="mean")


# --------------------------------------------------------------------------- #
# 3. The accounting downstream of the changed rate must follow
# --------------------------------------------------------------------------- #
def test_budget_still_balances_after_the_pe_penalty():
    """``a + c + u == K`` and ``u > 0`` -- the split must not silently break."""
    rd = run()["rounds"][0]
    assert rd.a_bits + rd.c_bits + rd.u_bits == pytest.approx(rd.k_bits)
    assert rd.feasible
    assert rd.u_bits > 0


def test_message_count_is_independent_of_eps_pe():
    """667 messages come from the *design* PE sample size, not from the rate.

    If this ever changes, the auth budget and the rate have become coupled and
    the "no circular dependency" argument in docs/11 §7.1 no longer holds.
    """
    counts = {run(eps_pe=e)["rounds"][0].extras["n_messages"]
              for e in (None, 1e-9, 1e-12)}
    assert len(counts) == 1


def test_k0_demand_does_not_change_with_eps_pe():
    """K_0 covers auth + seed, neither of which depends on the PE bound."""
    r_strict = run()
    r_nominal = run(eps_pe=None)
    assert r_strict["rounds"][0].a_bits == pytest.approx(
        r_nominal["rounds"][0].a_bits)
    assert r_strict["rounds"][0].c_bits == pytest.approx(
        r_nominal["rounds"][0].c_bits)
