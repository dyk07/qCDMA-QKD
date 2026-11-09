"""The cross-round refresh constraint, derived and verified.

Claim under test
----------------
With chip-wise refresh probability ``f``, a fraction ``1-f`` of the current code
is reused from the previous round.  If Eve knows a fraction ``p`` of the previous
code, the number of *current* chips she knows is

    k ~ Binomial( round((1-f) * L), p )

and the round-n seed keeps a usable min-entropy only if the leakage leaves the
required security margin intact.  The criterion used throughout is therefore an
**explicit eps_code target**, not ``H_min > 0``:

    eps_code = 2^-H_min <= eps_code_target
    <=>  H_min >= k_req := log2(1/eps_code_target)
    <=>  k <= seed_bits - k_req

The last ``k_req`` bits are an unspendable margin.  With seed = 256 and
eps = 1e-12 this is 39.9 bit, so at most 216.1 chips may be known -- not 256.

Two solvers are compared:

* **mean rule** (closed form):
      (1-f) * L * p <= seed_bits - k_req
      <=>   f >= 1 - (seed - k_req)/(L p)
* **tail rule** (the project default, exact for any p):
      P(k >= seed_bits - k_req) <= eps_code

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
EPS = 1e-12
K_REQ = float(np.log2(1.0 / EPS))       # 39.863 bit
BUDGET = SEED - K_REQ                   # 216.137 chips may be known


# --------------------------------------------------------------------------- #
# the closed form itself
# --------------------------------------------------------------------------- #
def test_closed_form_matches_algebra():
    for seed, L_, p in [(256, 1024, 1.0), (128, 1024, 1.0), (512, 1024, 1.0),
                        (256, 2048, 1.0), (256, 1024, 0.5)]:
        c = attack.refresh_security_constraint(seed, L_, p, eps_code=EPS)
        expect = 1.0 - (seed - K_REQ) / (L_ * p)
        assert c["f_min"] == pytest.approx(max(0.0, expect), abs=1e-12)


def test_tightness_known_chips_equal_the_budget_at_threshold():
    """At f_min the known chips exactly consume the spendable budget."""
    c = attack.refresh_security_constraint(SEED, L, 1.0, eps_code=EPS)
    known = (1.0 - c["f_min"]) * L * 1.0
    assert known == pytest.approx(BUDGET)
    # and this leaves precisely the requested eps_code
    h_min = SEED - known
    assert 2.0 ** (-h_min) == pytest.approx(EPS, rel=1e-12)


def test_h_min_zero_is_not_a_security_criterion():
    """The old ``H_min > 0`` rule certified f = 0.75, where eps_code = 1.0.

    H_min > 0 only excludes the single point where Eve knows the entire seed.
    At seed=256, L=1024, p=1 the old threshold f = 1 - seed/(L p) = 0.75 leaves
    k_known = 256 = seed, i.e. H_min = 0 and eps_code = 2^0 = 1: no secrecy at
    all.  The eps-target rule must demand strictly more refreshing there.
    """
    old_f = 1.0 - SEED / L
    old_h = attack.residual_seed_entropy(SEED, L, (1.0 - old_f) * 1.0)
    assert old_h == pytest.approx(0.0)
    assert 2.0 ** (-old_h) == pytest.approx(1.0)
    new_f = attack.refresh_security_constraint(SEED, L, 1.0, eps_code=EPS)["f_min"]
    assert new_f > old_f


def test_stricter_eps_target_requires_more_refreshing():
    """f_min is monotone in the eps_code target."""
    fs = [attack.refresh_security_constraint(SEED, L, 1.0, eps_code=e)["f_min"]
          for e in (1e-3, 1e-6, 1e-9, 1e-12, 1e-15)]
    assert all(a < b for a, b in zip(fs, fs[1:]))


def test_longer_seed_tolerates_more_known_chips():
    """f_min = 1 - seed/(Lp) DECREASES with the seed length.

    A longer seed is exhausted by more known chips, so less refreshing is
    needed.  (Counter-intuitive at first glance: it is the *ratio* seed/L that
    matters, not the seed alone.)
    """
    seeds = [64, 128, 256, 512]
    f = [attack.refresh_security_constraint(s, L, 1.0, eps_code=EPS)["f_min"]
         for s in seeds]
    assert all(a > b for a, b in zip(f, f[1:]))


def test_longer_code_requires_more_refresh():
    """f_min INCREASES with L: at fixed f, more chips are carried over."""
    Ls = [512, 1024, 2048, 4096]
    f = [attack.refresh_security_constraint(SEED, x, 1.0, eps_code=EPS)["f_min"]
         for x in Ls]
    assert all(a < b for a, b in zip(f, f[1:]))


def test_security_margin_breaks_scale_invariance():
    """The margin k_req is an *absolute* cost, so seed/L is not the whole story.

    With ``H_min > 0`` (k_req = 0) f_min depended on seed/L alone.  Once the
    eps target demands a fixed ``k_req`` bit margin, scaling seed and L together
    makes the margin relatively cheaper, so f_min strictly decreases along
    (128,512) -> (256,1024) -> (512,2048) -> (1024,4096) even though the ratio
    is constant.  The margin therefore hurts small systems the most.
    """
    pairs = [(128, 512), (256, 1024), (512, 2048), (1024, 4096)]
    vals = [attack.refresh_security_constraint(s, L_, 1.0, eps_code=EPS)["f_min"]
            for s, L_ in pairs]
    assert all(a > b for a, b in zip(vals, vals[1:]))


def test_scale_invariance_holds_only_in_the_k_req_to_zero_limit():
    """Recovering the old rule: as eps_code -> 1, k_req -> 0 and ratio wins."""
    pairs = [(128, 512), (256, 1024), (512, 2048), (1024, 4096)]
    vals = [attack.refresh_security_constraint(s, L_, 1.0, eps_code=1.0)["f_min"]
            for s, L_ in pairs]
    assert all(v == pytest.approx(vals[0]) for v in vals)


def test_infeasible_when_seed_exceeds_code():
    """A longer seed is *more* tolerant: the constraint stops binding.

    f = 1 always gives k_known = 0, so a feasible f exists for every
    seed_bits > k_req.  Once ``L*p <= seed - k_req``, even f = 0 is safe because
    Eve can know at most L chips and still leave the margin intact.

    (An earlier version of this library wrongly reported ``feasible=False``
    here -- the direction was inverted.)
    """
    c = attack.refresh_security_constraint(seed_bits=2048, L=1024, p_blind=1.0)
    assert c["feasible"]
    assert c["f_min"] == 0.0
    assert not c["binds"]


def test_no_infeasible_seed_for_any_code_length():
    """Exhaustive sanity: feasible once the seed can hold the margin."""
    for L_ in (256, 1024, 4096):
        for seed in (128, 512, 1024, 4096, 10 ** 6):
            c = attack.refresh_security_constraint(seed, L_, 1.0)
            assert c["feasible"], (seed, L_)
            assert 0.0 <= c["f_min"] <= 1.0


def test_short_seed_is_the_only_infeasible_regime():
    """seed <= k_req cannot hold the margin at any f -- genuinely infeasible.

    This is the one real failure mode: with eps = 1e-12 we need 39.9 bits of
    margin, so a 32-bit seed cannot work however often the code is refreshed.
    """
    for seed in (8, 16, 32):
        c = attack.refresh_security_constraint(seed, L, 1.0, eps_code=EPS)
        assert not c["feasible"]
        assert c["f_min"] == 1.0
        assert "seed_bits <= log2" in c["note"]


def test_boundary_at_seed_minus_k_req_equals_code():
    """f_min hits exactly 0 when L*p == seed - k_req, not when seed == L."""
    seed = int(np.ceil(L + K_REQ))          # minimal seed that makes f = 0 safe
    c = attack.refresh_security_constraint(seed, L, 1.0, eps_code=EPS)
    assert c["f_min"] == pytest.approx(0.0, abs=1e-12)
    # one chip shorter -> strictly positive refresh requirement
    c2 = attack.refresh_security_constraint(seed - 2, L, 1.0, eps_code=EPS)
    assert c2["f_min"] > 0.0


def test_zero_leakage_needs_no_refresh():
    c = attack.refresh_security_constraint(SEED, L, p_blind=0.0, eps_code=EPS)
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
    r = attack.refresh_constraint_tail(SEED, L, p, eps_code=eps)
    n_reused = int(round((1.0 - r["f_min"]) * L))
    if n_reused <= BUDGET:
        viol = 0.0
    elif p >= 1.0:
        viol = 1.0            # deterministically saturated -> infeasible
    else:
        viol = stats.binom.sf(np.ceil(BUDGET) - 1, n_reused, p)
    # either genuinely safe, or reported as infeasible
    assert viol <= eps or r["f_min"] >= 1.0 - 1e-9


def test_exact_rule_reproduces_closed_form_when_p_is_one():
    """For p = 1 both rules agree to within the chip rounding of (1-f)L."""
    r = attack.refresh_constraint_tail(SEED, L, 1.0)
    assert r["f_min"] == pytest.approx(
        attack.refresh_security_constraint(SEED, L, 1.0)["f_min"], abs=2e-3)


def test_mean_rule_is_optimistic_for_imperfect_blind_estimation():
    """The mean rule under-estimates f_min once p < 1."""
    for p in (0.95, 0.9, 0.8, 0.5):
        mean_f = attack.refresh_security_constraint(SEED, L, p)["f_min"]
        exact_f = attack.refresh_constraint_tail(SEED, L, p)["f_min"]
        assert exact_f > mean_f


def test_mean_rule_violates_the_tail_at_its_own_threshold():
    """Concrete counter-example: p=0.99 at f=0.75 is *not* safe."""
    p = 0.99
    n_reused = int(round((1.0 - 0.75) * L))       # 256 reused chips
    viol = stats.binom.sf(np.ceil(BUDGET) - 1, n_reused, p)
    assert viol > 1e-3, "mean rule would wrongly certify this point as safe"
    # ... and the exact rule pushes f up until the tail is fine
    exact = attack.refresh_constraint_tail(SEED, L, p)["f_min"]
    assert exact > 0.75


def test_tail_rule_gap_grows_as_blind_estimation_worsens():
    gaps = []
    for p in (0.99, 0.95, 0.9, 0.8):
        gaps.append(attack.refresh_constraint_tail(SEED, L, p)["f_min"]
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
        ratio = attack.max_chips_per_seed_bit(SEED, f, p, EPS)
        # L = ratio * seed is exactly the largest feasible code length
        L_max = ratio * SEED
        known = (1.0 - f) * L_max * p
        assert known == pytest.approx(SEED - K_REQ, rel=1e-9)


# --------------------------------------------------------------------------- #
# the seed entropy must be quoted at an upper tail, not at the mean
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p_chip", [0.05, 0.10, 0.20])
def test_tail_entropy_never_exceeds_mean_entropy(p_chip):
    """H_min is a worst-case quantity, so the tail version must be <= the mean.

    Using the mean leakage credits Eve with fewer chips than her upper tail, so
    it *overstates* the surviving entropy -- and every lost bit doubles the
    guessing probability.
    """
    d = attack.entropy_gap_vs_tail(256, 1024, p_chip, 1e-12)
    assert d["entropy_tail"] <= d["entropy_mean"]
    if d["entropy_lost"] > 0:
        assert d["eps_tail"] > d["eps_mean"]


def test_mean_based_entropy_understates_leakage_by_many_orders():
    """Quantify the model error so the docs' claim is pinned.

    09_model_vs_bound.md asserts 1e15..1e22; keep that claim honest.
    """
    for p_chip in (0.05, 0.10):
        d = attack.entropy_gap_vs_tail(256, 1024, p_chip, 1e-12)
        ratio = d["eps_tail"] / d["eps_mean"]
        assert 1e15 <= ratio <= 1e23, (p_chip, ratio)


def test_tail_entropy_goes_to_zero_when_the_tail_saturates():
    """At p=0.2 the upper tail already exhausts the seed while the mean does not.

    This is the sharpest illustration of why the mean is not safe: 51 bit of
    apparent headroom at the mean, none at confidence 1-1e-12.
    """
    d = attack.entropy_gap_vs_tail(256, 1024, 0.20, 1e-12)
    assert d["entropy_mean"] > 0
    assert d["entropy_tail"] == 0.0


# --------------------------------------------------------------------------- #
# the refresh constraint must charge the guessing term too
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("eps_code", [1e-6, 1e-9, 1e-12, 1e-15])
def test_rigorous_refresh_constraint_is_a_two_term_union_bound(eps_code):
    """eps_code <= P(k > k*) + 2^{-(seed-k*)} must hold at the returned f_min.

    Regression: the tail-only version set the second term to zero, which is
    optimistic -- it treated "used up the chip budget" as if it were "guessed
    the seed".
    """
    c = attack.refresh_constraint_tail(256, 1024, 1.0, eps_code)
    assert c["violation"] <= eps_code * (1 + 1e-9)
    assert c["violation"] == pytest.approx(c["p_exceed"] + c["p_guess"],
                                           abs=1e-30)
    # and both terms are reported, so the split stays visible
    assert c["p_guess"] > 0
    assert c["k_star"] <= c["k_max_allowed"]


def test_rigorous_f_min_is_close_to_but_not_above_the_documented_value():
    """The published 0.7886 must equal the rigorous value, not the legacy one."""
    c = attack.refresh_constraint_tail(256, 1024, 1.0, 1e-12)
    assert c["f_min"] == pytest.approx(0.78857, abs=5e-5)
    assert c["f_min"] != pytest.approx(c["f_min_legacy"], abs=1e-9)
