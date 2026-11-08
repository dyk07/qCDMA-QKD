"""The cross-round refresh constraint, derived and verified.

Claim under test
----------------
With chip-wise refresh probability ``f``, a fraction ``1-f`` of the current code
is reused from the previous round.  If Eve knows a fraction ``p`` of the previous
code, the number of *current* chips she knows is

    k ~ Binomial( round((1-f) * L), p )

and the round-n seed keeps positive min-entropy only if ``k < seed_bits``.

Two rules are compared here:

* **mean rule** (closed form, what the docs quote):
      (1-f) * L * p < seed_bits   <=>   f > 1 - seed/(L p)
* **tail rule** (correct whenever p < 1):
      P(k >= seed_bits) <= eps_code

They agree to within one chip when ``p = 1`` and diverge once blind estimation
is imperfect; the mean rule is then *optimistic*, because it constrains the
average rather than the upper tail.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from qcdma_qkd import attack

L = 1024
SEED = 256


# --------------------------------------------------------------------------- #
# the closed form itself
# --------------------------------------------------------------------------- #
def test_closed_form_matches_algebra():
    for seed, L_, p in [(256, 1024, 1.0), (128, 1024, 1.0), (512, 1024, 1.0),
                        (256, 2048, 1.0), (256, 1024, 0.5)]:
        c = attack.refresh_security_constraint(seed, L_, p)
        assert c["f_min"] == pytest.approx(1.0 - seed / (L_ * p), abs=1e-12)


def test_tightness_known_chips_equal_seed_at_threshold():
    """At f_min the number of known chips is exactly the seed length."""
    c = attack.refresh_security_constraint(SEED, L, 1.0)
    known = (1.0 - c["f_min"]) * L * 1.0
    assert known == pytest.approx(SEED)


def test_longer_seed_tolerates_more_known_chips():
    """f_min = 1 - seed/(Lp) DECREASES with the seed length.

    A longer seed is exhausted by more known chips, so less refreshing is
    needed.  (Counter-intuitive at first glance: it is the *ratio* seed/L that
    matters, not the seed alone.)
    """
    seeds = [64, 128, 256, 512]
    f = [attack.refresh_security_constraint(s, L, 1.0)["f_min"] for s in seeds]
    assert all(a > b for a, b in zip(f, f[1:]))


def test_longer_code_requires_more_refresh():
    """f_min INCREASES with L: at fixed f, more chips are carried over."""
    Ls = [512, 1024, 2048, 4096]
    f = [attack.refresh_security_constraint(SEED, x, 1.0)["f_min"] for x in Ls]
    assert all(a < b for a, b in zip(f, f[1:]))


def test_only_the_ratio_seed_over_L_matters():
    """f_min depends on seed/L alone (at p = 1)."""
    pairs = [(128, 512), (256, 1024), (512, 2048), (1024, 4096)]
    vals = [attack.refresh_security_constraint(s, L_, 1.0)["f_min"]
            for s, L_ in pairs]
    assert all(v == pytest.approx(vals[0]) for v in vals)


def test_infeasible_when_seed_exceeds_code():
    """A longer seed is *more* tolerant: the constraint stops binding.

    f = 1 always gives k_known = 0, so a feasible f exists for every
    seed_bits > 0.  Once seed >= L*p, even f = 0 is safe because Eve can know
    at most L chips, which is no more than the seed length.

    (An earlier version of this library wrongly reported ``feasible=False``
    here -- the direction was inverted.)
    """
    c = attack.refresh_security_constraint(seed_bits=2048, L=1024, p_blind=1.0)
    assert c["feasible"]
    assert c["f_min"] == 0.0
    assert not c["binds"]


def test_no_infeasible_seed_for_any_code_length():
    """Exhaustive sanity: never infeasible, whatever the seed/code ratio."""
    for L_ in (256, 1024, 4096):
        for seed in (32, 128, 512, 1024, 4096, 10 ** 6):
            c = attack.refresh_security_constraint(seed, L_, 1.0)
            assert c["feasible"], (seed, L_)
            assert 0.0 <= c["f_min"] <= 1.0


def test_boundary_seed_equals_code():
    """seed == L is the tipping point: f_min hits exactly 0."""
    c = attack.refresh_security_constraint(seed_bits=1024, L=1024, p_blind=1.0)
    assert c["f_min"] == pytest.approx(0.0)
    # one chip shorter seed -> strictly positive refresh requirement
    c2 = attack.refresh_security_constraint(seed_bits=1023, L=1024, p_blind=1.0)
    assert c2["f_min"] > 0.0


def test_zero_leakage_needs_no_refresh():
    c = attack.refresh_security_constraint(SEED, L, p_blind=0.0)
    assert c["f_min"] == 0.0
    assert c["feasible"]


def test_prf_model_has_no_constraint():
    assert attack.refresh_security_constraint(SEED, L, model="prf")["f_min"] == 0.0


# --------------------------------------------------------------------------- #
# the tail rule is the one that is actually correct
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p", [1.0, 0.999, 0.99, 0.95, 0.9, 0.8, 0.5])
def test_exact_rule_satisfies_its_own_tail(p):
    """At the returned f_min the violation probability must meet eps_code."""
    eps = 1e-12
    r = attack.refresh_constraint_exact(SEED, L, p, eps_code=eps)
    n_reused = int(round((1.0 - r["f_min"]) * L))
    if n_reused < SEED:
        viol = 0.0
    elif p >= 1.0:
        viol = 1.0            # deterministically saturated -> infeasible
    else:
        viol = stats.binom.sf(SEED - 1, n_reused, p)
    # either genuinely safe, or reported as infeasible
    assert viol <= eps or r["f_min"] >= 1.0 - 1e-9


def test_exact_rule_reproduces_closed_form_when_p_is_one():
    """For p = 1 both rules agree to within the chip rounding of (1-f)L."""
    r = attack.refresh_constraint_exact(SEED, L, 1.0)
    assert r["f_min"] == pytest.approx(
        attack.refresh_security_constraint(SEED, L, 1.0)["f_min"], abs=2e-3)


def test_mean_rule_is_optimistic_for_imperfect_blind_estimation():
    """The mean rule under-estimates f_min once p < 1."""
    for p in (0.95, 0.9, 0.8, 0.5):
        mean_f = attack.refresh_security_constraint(SEED, L, p)["f_min"]
        exact_f = attack.refresh_constraint_exact(SEED, L, p)["f_min"]
        assert exact_f > mean_f


def test_mean_rule_violates_the_tail_at_its_own_threshold():
    """Concrete counter-example: p=0.99 at f=0.75 is *not* safe."""
    p = 0.99
    n_reused = int(round((1.0 - 0.75) * L))       # 256 reused chips
    viol = stats.binom.sf(SEED - 1, n_reused, p)
    assert viol > 1e-3, "mean rule would wrongly certify this point as safe"
    # ... and the exact rule pushes f up until the tail is fine
    exact = attack.refresh_constraint_exact(SEED, L, p)["f_min"]
    assert exact > 0.75


def test_tail_rule_gap_grows_as_blind_estimation_worsens():
    gaps = []
    for p in (0.99, 0.95, 0.9, 0.8):
        gaps.append(attack.refresh_constraint_exact(SEED, L, p)["f_min"]
                    - attack.refresh_security_constraint(SEED, L, p)["f_min"])
    assert all(a < b for a, b in zip(gaps, gaps[1:]))


# --------------------------------------------------------------------------- #
# consistency with the leakage model
# --------------------------------------------------------------------------- #
def test_eps_code_matches_remaining_entropy():
    """cross_round_eps must equal 2^-H_min with H_min from residual_seed_entropy."""
    for f in (0.0, 0.5, 0.75, 0.9, 1.0):
        p = 1.0
        e = attack.cross_round_eps(f, SEED, L, p)
        h = attack.residual_seed_entropy(SEED, L, (1.0 - f) * p)
        assert e == pytest.approx(2.0 ** (-h))


def test_full_refresh_leaves_full_entropy():
    assert attack.residual_seed_entropy(SEED, L, 0.0) == pytest.approx(SEED)


def test_no_refresh_with_full_known_code_is_fully_exposed():
    assert attack.residual_seed_entropy(SEED, L, 1.0) == pytest.approx(0.0)


def test_max_chips_per_seed_bit_inverts_the_constraint():
    for f, p in [(0.75, 1.0), (0.5, 1.0), (0.9, 0.8)]:
        ratio = attack.max_chips_per_seed_bit(SEED, f, p)
        # L = ratio * seed is exactly the largest feasible code length
        L_max = ratio * SEED
        known = (1.0 - f) * L_max * p
        assert known == pytest.approx(SEED, rel=1e-9)
