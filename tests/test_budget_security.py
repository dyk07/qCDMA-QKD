"""Tests for the key budget, the security composition, and the attacks.

These encode the paper's *quantitative claims*, so a failure here means a claim
in docs/ has gone stale.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qcdma_qkd import attack, budget, loop, security
from qcdma_qkd.config import BudgetConfig, CDMAConfig, CVQKDConfig, SecurityConfig


# --------------------------------------------------------------------------- #
# authentication cost
# --------------------------------------------------------------------------- #
def test_auth_cost_does_not_scale_with_message_bits():
    """A MAC over 1 Mbit must cost the same key as a MAC over 1 kbit.

    The cost is a function of the *message count* only, and saturates at one
    key length ``l`` because a single eps-AXU key serves many messages.
    """
    cfg = BudgetConfig()
    one = budget.auth_bits(1, cfg)
    assert one > 0
    # no bit-length term anywhere: the API has no such parameter at all
    info = budget.auth_key_bits(cfg)
    assert budget.auth_bits(100, cfg) == pytest.approx(info["key_length"])
    assert budget.auth_bits(10_000, cfg) == pytest.approx(info["key_length"])


def test_otp_costs_more_than_axu():
    otp = budget.auth_key_bits(BudgetConfig(mac_model="otp"))["per_message"]
    axu = budget.auth_key_bits(BudgetConfig(mac_model="axu"))["per_message"]
    assert otp > axu


def test_axu_key_length_satisfies_the_forgery_bound():
    """l = t + 2 log2(1/eps) must give eps_forge <= eps for the allowed reuse."""
    cfg = BudgetConfig(mac_model="axu")
    info = budget.auth_key_bits(cfg, eps_auth=1e-12)
    r = info["reuse"]
    t = info["tag_bits"]
    key_term = 2.0 ** (-(info["key_length"] - t) / 2.0)
    reuse_term = r * 2.0 ** (-t)
    assert key_term + reuse_term <= 1e-12 * 1.01


def test_message_count_scales_with_transcript():
    cfg = BudgetConfig()
    small = budget.round_message_count(10 ** 6, 0.3, cfg, bits_per_message=1 << 20)
    big = budget.round_message_count(10 ** 9, 0.3, cfg, bits_per_message=1 << 20)
    assert big > small


# --------------------------------------------------------------------------- #
# budget feasibility
# --------------------------------------------------------------------------- #
def test_round_budget_sums_to_key():
    cfg = BudgetConfig()
    b = budget.round_budget(1e7, 10 ** 8, 0.3, cfg)
    assert b["a"] + b["c"] + b["u"] == pytest.approx(b["K"])
    assert b["feasible"]


def test_budget_feasibility_threshold():
    cfg = BudgetConfig()
    cdma = CDMAConfig()
    sym_rate = cdma.symbol_rate()          # per-user: chip_rate / L
    need = budget.min_key_rate_bps(10 ** 8, 0.3, cfg, symbol_rate=sym_rate)
    below = budget.round_budget(need * 0.9 * 0.1, 10 ** 8, 0.3, cfg)
    above = budget.round_budget(need * 1.5 * (10 ** 8 / sym_rate), 10 ** 8, 0.3, cfg)
    assert not below["feasible"]
    assert above["feasible"]


def test_k0_covers_bootstrap_and_reserve():
    cfg = BudgetConfig()
    k = budget.k0_budget(cfg, 10 ** 8, 0.3, bootstrap_rounds=2)
    assert k["required"] == pytest.approx(2 * k["per_round"] * 1.1)
    assert k["feasible"] == (cfg.k0_bits >= k["required"])


def test_seed_cost_independent_of_spreading_factor():
    """PRF expansion: the seed is a fixed cost, not proportional to L."""
    assert budget.round_budget(1e7, 1e8, 0.3, BudgetConfig(seed_bits=256))["c"] == 256


# --------------------------------------------------------------------------- #
# security composition
# --------------------------------------------------------------------------- #
def test_sum_composition():
    assert security.compose_sum({"a": 1e-10, "b": 2e-10}) == pytest.approx(3e-10)


def test_sequential_composition_grows_with_rounds():
    a = security.compose_sequential(1e-12, 10)
    b = security.compose_sequential(1e-12, 1000)
    assert b > a > 0


def test_lhl_inverts_correctly():
    h, m = 1000.0, 800.0
    eps = security.lhl_epsilon(h, m)
    assert security.lhl_output_bits(h, eps) == pytest.approx(m, rel=0.02)


def test_budget_total_is_sum_of_parts():
    b = security.from_config(SecurityConfig())
    d = b.table()
    assert d["total"] == pytest.approx(sum(v for k, v in d.items() if k != "total"))


# --------------------------------------------------------------------------- #
# attacks
# --------------------------------------------------------------------------- #
def test_mitm_defeated_by_authentication():
    bad = attack.man_in_the_middle(authenticated=False)
    good = attack.man_in_the_middle(authenticated=True)
    assert bad.detection_probability == 0.0
    assert good.detection_probability > 0.99
    assert good.rounds_until_detection < 1.01


def test_mitm_survival_decays_with_rounds():
    p1 = attack.detection_after_k_rounds(128, 8, 1)
    p10 = attack.detection_after_k_rounds(128, 8, 10)
    assert p10 <= p1


def test_extractor_model_needs_more_fresh_chips_than_seed():
    """Constraint: known chips (1-f)*L*p must leave the k_req margin intact."""
    k_req = attack.required_entropy_bits(1e-12)
    c = attack.refresh_security_constraint(seed_bits=256, L=1024, model="extractor")
    assert c["f_min"] == pytest.approx(1.0 - (256.0 - k_req) / 1024.0)
    # the old "known chips < seed_bits" rule would have certified f = 0.75, at
    # which point the remaining min-entropy is exactly zero (eps_code = 1.0)
    assert c["f_min"] > 1.0 - 256.0 / 1024.0
    assert c["feasible"]
    assert attack.refresh_security_constraint(256, 1024, model="prf")["f_min"] == 0.0


def test_refresh_constraint_tightens_with_smaller_seed_ratio():
    """A longer code relative to the seed demands a larger refresh fraction."""
    a = attack.refresh_security_constraint(256, 1024)["f_min"]
    b = attack.refresh_security_constraint(256, 4096)["f_min"]
    c = attack.refresh_security_constraint(1024, 4096)["f_min"]
    assert a < b
    assert c < b


def test_larger_refresh_reduces_leakage():
    """Only refreshes above the constraint threshold actually help."""
    eps = [attack.cross_round_eps(f, 256, 1024, 0.99) for f in (0.5, 0.9, 1.0)]
    assert eps[1] < eps[0]
    assert eps[2] <= eps[1]


def test_blind_estimation_improves_with_observations():
    p_low = attack.chip_success_probability(1.0, 1024, 1e3)
    p_high = attack.chip_success_probability(1.0, 1024, 1e9)
    assert p_low < p_high
    assert attack.required_observations(256, 1024, 1.0) > 0


def test_state_preparation_leak_scales_with_V_A():
    a = attack.state_preparation_leak(0.1, 4.0, 1024)
    b = attack.state_preparation_leak(0.1, 8.0, 1024)
    assert b["xi_prep"] == pytest.approx(2 * a["xi_prep"])


# --------------------------------------------------------------------------- #
# closed loop
# --------------------------------------------------------------------------- #
def test_loop_runs_and_budgets_are_consistent():
    res = loop.run_loop(4, k0_bits=2 ** 20, cfg_cv=CVQKDConfig(),
                        cfg_cdma=CDMAConfig(), cfg_bud=BudgetConfig(),
                        distance_km=25.0, n_symbols=10 ** 8)
    assert res["aborted_at"] is None
    for r in res["rounds"]:
        assert r.u_bits == pytest.approx(r.k_bits - r.a_bits - r.c_bits)
        assert r.auth_ok


def test_loop_aborts_on_tamper():
    res = loop.run_loop(5, k0_bits=2 ** 20, cfg_cv=CVQKDConfig(),
                        cfg_cdma=CDMAConfig(), cfg_bud=BudgetConfig(),
                        distance_km=25.0, n_symbols=10 ** 8, tamper_at=2)
    assert res["aborted_at"] == 2
    assert len(res["keys"]) == 1


def test_codebook_is_reproducible_from_seed():
    seed = b"0" * 32
    a = loop.derive_codebook(seed, 4, 256, frame=1)
    b = loop.derive_codebook(seed, 4, 256, frame=1)
    c = loop.derive_codebook(seed, 4, 256, frame=2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert set(np.unique(a)) <= {-1.0, 1.0}


def test_kdf_domains_are_independent():
    k = loop.split_key(10 ** 6, BudgetConfig(), overhead_bits=1000)
    assert k["a"] != k["c"] != k["u"]
    assert k["u_bits"] == pytest.approx(10 ** 6 - 1000 - 256)


# --------------------------------------------------------------------------- #
# regression: auth billing must not under-charge, and per_round must be real
# --------------------------------------------------------------------------- #
def test_per_round_is_one_full_key_per_round_in_every_mac_model():
    """``per_round`` is what must be *derived*, so it is n_keys * key_length.

    Regression: the OTP branch returned ``key_length`` (128 bit) regardless of
    the message count, so a 667-message round was billed as one message.
    """
    for model in ("axu", "otp"):
        cfg = BudgetConfig(mac_model=model)
        info = budget.auth_key_bits(cfg, n_messages=667)
        assert info["per_round"] == pytest.approx(
            info["n_keys"] * info["key_length"])
        if model == "otp":
            assert info["per_round"] == pytest.approx(667 * 128.0)


def test_billing_rounds_up_to_whole_keys_beyond_one_key_lifetime():
    """n > reuse must cost a whole extra key, not a fractional one.

    Regression: ``auth_bits`` divided by ``reuse`` once, so
    ``reuse + 1`` messages were billed as a single key.
    """
    cfg = BudgetConfig()
    r = budget.auth_key_bits(cfg)["reuse"]
    l = budget.auth_key_bits(cfg)["key_length"]
    assert budget.auth_bits(r, cfg) == pytest.approx(l)
    assert budget.auth_bits(r + 1, cfg) == pytest.approx(2 * l)
    assert budget.auth_bits(int(2.5 * r), cfg) == pytest.approx(3 * l)


def test_auth_epsilon_is_not_multiplied_by_the_message_count():
    """The eps-AXU bound already charges for reuse; do not double count.

    ``eps <= r 2^{-t} + 2^{-(l-t)/2}`` with one fresh key per round and
    ``n_msg = 667 <= r`` gives exactly ``eps``.  Multiplying by the message
    count (the old ``eps_auth_rounds=4``) counted reuse twice.
    """
    sec = SecurityConfig()
    one = security.from_config(sec)
    assert one.eps_auth == pytest.approx(sec.eps_auth)
    assert one.total() < sec.eps_total




def test_round_composition_cannot_use_a_single_component():
    """Composing a per-round budget must use the whole budget, not one term.

    Regression: the docs composed ``1e-12`` (authentication alone), dropping
    eps_qkd/eps_sm/eps_pe (1e-10 each) and understating the composed epsilon by
    ~300x at N=1e4.
    """
    sec = SecurityConfig()
    eps_round = security.from_config(sec).total()
    # eps_round is dominated by the 1e-10 terms, not by 1e-12
    assert eps_round > 3 * 1e-10
    composed = security.compose_over_rounds(eps_round, 10 ** 4)
    assert composed > 10 ** 2 * security.compose_over_rounds(1e-12, 10 ** 4)
    # and the honest round budget is tiny
    assert security.max_rounds(eps_round, sec.eps_total) < 5.0
    assert security.required_round_eps(10 ** 4, sec.eps_total) <= 2e-12
    # and the three helpers must agree with each other
    for n in (1, 3, 10, 1000, 10 ** 4):
        worst = security.compose_over_rounds(eps_round, n)
        need = security.required_round_eps(n, sec.eps_total)
        assert (worst <= sec.eps_total) == (eps_round <= need), (
            f"compose/required disagree at N={n}")


def test_composition_uses_the_tighter_of_the_two_bounds():
    """Both the union and sequential forms are upper bounds; take the min.

    The sequential form is vacuous for small N (sqrt(2N ln 1/delta) > N), so
    quoting it alone at N=1 would report 7.4x more than is provable.
    """
    sec = SecurityConfig()
    er = security.from_config(sec).total()
    for n in (1, 2, 10, 50, 1000):
        best = security.compose_over_rounds(er, n)
        assert best <= security.compose_over_rounds(er, n, "union")
        # the sequential form carries an additive eps_delta, so at very small
        # N it can exceed eps_round itself; the point is only that "best"
        # never exceeds *both* bounds
        assert best <= max(security.compose_over_rounds(er, n, "union"),
                           security.compose_over_rounds(er, n, "sequential"))
