"""The code model: PRF (computational) vs extractor (information-theoretic).

The project's final answer for the sample scenario is

    code_model = "prf",  refresh_fraction = 1.0

because a PRF makes chip leakage free of seed-leakage, so no refresh constraint
exists, and ``f = 1`` additionally removes cross-round correlation.  The price
is that the code layer is then **computationally** secret, protected by a PRF
assumption rather than by an extractor argument.

The information-theoretic alternative (``code_model = "extractor"``) is kept
fully implemented: a 256-bit seed cannot absorb ``L = 1024`` known chips, so it
demands ``f >= f_min`` (docs/05 §5).  These tests pin both paths and the
boundary between them.
"""

from __future__ import annotations

import numpy as np
import pytest

from qcdma_qkd import codes, loop
from qcdma_qkd.config import BudgetConfig, CDMAConfig, CVQKDConfig

L = 1024
SEED = 256
EPS = 1e-12


# --------------------------------------------------------------------------- #
# the default IS the sample model
# --------------------------------------------------------------------------- #
def test_default_is_prf_with_full_refresh():
    """The shipped default must be exactly the chosen sample model."""
    c = CDMAConfig()
    assert c.code_model == "prf"
    assert c.refresh_fraction == 1.0
    assert not c.is_information_theoretic()


def test_prf_has_no_refresh_constraint():
    c = CDMAConfig()
    r = c.refresh_constraint(SEED, eps_code=EPS)
    assert r["f_min"] == 0.0
    assert r["satisfied"]
    assert r["security"] == "computational"


def test_prf_full_refresh_is_satisfied_by_construction():
    """f = 1 is the most conservative choice and always allowed under a PRF."""
    for f in (1.0, 0.5, 0.0):
        c = CDMAConfig(code_model="prf", refresh_fraction=f)
        assert c.refresh_constraint(SEED, eps_code=EPS)["satisfied"]


# --------------------------------------------------------------------------- #
# the extractor alternative is real, and binding
# --------------------------------------------------------------------------- #
def test_extractor_constraint_matches_the_tail_solver():
    from qcdma_qkd import attack

    c = CDMAConfig(code_model="extractor")
    r = c.refresh_constraint(SEED, eps_code=EPS)
    ref = attack.refresh_constraint_tail(SEED, L, 1.0, eps_code=EPS)
    assert r["f_min"] == pytest.approx(ref["f_min"])
    # 0.7886 for the defaults -- not 0, and not the old 0.75
    assert 0.78 < r["f_min"] < 0.80


def test_extractor_f1_is_always_satisfied():
    """f = 1 removes every reused chip, so it must pass for any seed > k_req."""
    c = CDMAConfig(code_model="extractor", refresh_fraction=1.0)
    assert c.refresh_constraint(SEED, eps_code=EPS)["satisfied"]


def test_extractor_rejects_the_old_h_min_rule_threshold():
    """f = 0.75 was the H_min>0 answer; it must now be rejected.

    At f = 0.75 the reused chips exactly equal the seed, so the remaining
    min-entropy is 0 and the achievable eps_code is 1.0.
    """
    c = CDMAConfig(code_model="extractor", refresh_fraction=0.75)
    r = c.refresh_constraint(SEED, eps_code=EPS)
    assert not r["satisfied"]
    assert r["f_min"] > 0.75


def test_extractor_check_refresh_raises_on_violation():
    c = CDMAConfig(code_model="extractor", refresh_fraction=0.5)
    with pytest.raises(ValueError, match="violates the extractor constraint"):
        c.check_refresh(SEED, eps_code=EPS)


def test_check_refresh_passes_for_the_default():
    assert CDMAConfig().check_refresh(SEED, eps_code=EPS)["satisfied"]


def test_bad_code_model_is_rejected():
    with pytest.raises(ValueError, match="code_model must be"):
        CDMAConfig(code_model="magic")


def test_bad_refresh_fraction_is_rejected():
    with pytest.raises(ValueError, match="refresh_fraction must be"):
        CDMAConfig(refresh_fraction=1.5)


# --------------------------------------------------------------------------- #
# the loop must actually honour refresh_fraction
# --------------------------------------------------------------------------- #
CV = CVQKDConfig()
BD = BudgetConfig()


def _two_round_codebooks(f: float):
    cfg = CDMAConfig(refresh_fraction=f)
    res = loop.run_loop(2, 2 ** 20, CV, cfg, BD, 25.0, n_symbols=10 ** 5)
    return res["rounds"][0].codebook, res["rounds"][1].codebook


@pytest.mark.parametrize("f", [1.0, 0.75, 0.5, 0.0])
def test_cross_round_correlation_equals_one_minus_f(f):
    """<c_n, c_{n-1}>/L == 1 - f: this is what makes the constraint meaningful."""
    c_prev, c_cur = _two_round_codebooks(f)
    measured = float(np.mean([
        codes.normalised_cross_correlation(c_prev[i], c_cur[i])
        for i in range(c_prev.shape[0])
    ]))
    assert measured == pytest.approx(1.0 - f, abs=0.02)


def test_full_refresh_decorrelates_rounds():
    c_prev, c_cur = _two_round_codebooks(1.0)
    measured = float(np.mean([
        codes.normalised_cross_correlation(c_prev[i], c_cur[i])
        for i in range(c_prev.shape[0])
    ]))
    assert abs(measured) < 0.03, "f=1 must leave no cross-round correlation"


def test_no_refresh_carries_the_code_verbatim():
    c_prev, c_cur = _two_round_codebooks(0.0)
    assert np.array_equal(c_prev, c_cur)


def test_loop_summary_reports_the_code_model():
    res = loop.run_loop(2, 2 ** 20, CV, CDMAConfig(), BD, 25.0, n_symbols=10 ** 5)
    s = loop.loop_summary(res)
    assert s["code_model"] == "prf"
    assert s["refresh_fraction"] == 1.0
