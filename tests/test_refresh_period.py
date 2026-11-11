"""The PRF seed-refresh period ``T`` as a three-way trade-off.

The knob
--------
``T`` = symbols between PRF re-draws of the codebook from the seed.  One
parameter trades three objectives, each with a different dependence:

===========  ================================  =====================
axis         quantity                          scaling in ``T``
===========  ================================  =====================
security     samples per codebook              ``T`` (needs ``T < n_th``)
MUI          rel. std of block-averaged xi     ``sqrt(T/B)`` (small T better)
PRF          bits per symbol                   ``1/T`` (small T worse)
===========  ================================  =====================

The security and PRF legs pull in **opposite directions**; the MUI leg agrees
with security.  So there is a feasible interval, not an optimum:

    T <= min(n_th(L), T_PRF_budget)

with ``T_PRF_budget = M * R_c / (PRF capacity in bit/s)``.

These tests pin the *shape* of each leg and the direction of each trade.  The
transition point itself is measured in ``experiments/e12_refresh_period.py``,
because no honest closed form exists for it.
"""

from __future__ import annotations

import numpy as np
import pytest

from qcdma_qkd import attack, mui

L = 1024
M = 8                       # total users incl. victim
KAPPA = np.ones(M)          # powers incl. victim (for the attack legs)
KAPPA_INT = np.ones(M - 1)  # interferers only (for the MUI legs)
BLOCK = 10 ** 8
CHIP_RATE = 1.0e9


def trade(ts):
    """Attack-side trade-off: ``kappa`` includes the victim."""
    return attack.three_way_tradeoff(L, KAPPA, ts, block_symbols=BLOCK,
                                     chip_rate=CHIP_RATE)


def trade_mui(ts):
    """MUI leg: pass all users; the helper slices ``M-1`` interferers itself.

    ``kappa`` must include the victim because the *attack* leg does, so the MUI
    leg slices ``kappa[:M-1]``.  Relying on that default keeps the two legs
    consistent with ``04_mui_analysis.md`` (``sqrt(2/K)``, K = 7).
    """
    return attack.three_way_tradeoff(L, KAPPA, ts, block_symbols=BLOCK,
                                     chip_rate=CHIP_RATE)


# --------------------------------------------------------------------------- #
# 1. The security leg: a threshold in T, not a smooth curve
# --------------------------------------------------------------------------- #
def test_security_leg_is_a_threshold_in_t():
    """``blocked`` flips exactly once, at ``T = n_th``, and never before."""
    d = trade([1, 1e3, 1e5, 1e6, 1e8])
    n_th = d["n_threshold"]
    for r in d["rows"]:
        assert r["attack_blocked"] == (r["T"] < n_th)
    flags = [r["attack_blocked"] for r in d["rows"]]
    assert flags == sorted(flags, reverse=True)


def test_security_margin_scales_linearly_in_t():
    """``margin = T / n_th``, and ``n_th`` itself does not depend on T."""
    d = trade([100, 1000, 10_000])
    margins = [r["margin"] for r in d["rows"]]
    assert margins[1] == pytest.approx(10 * margins[0], rel=1e-9)
    assert margins[2] == pytest.approx(100 * margins[0], rel=1e-9)
    assert all(r["n_threshold"] == pytest.approx(d["n_threshold"])
               for r in d["rows"])


def test_block_caps_the_samples_a_single_codebook_can_supply():
    """A refresh period longer than the block is just a frozen codebook."""
    d = trade([BLOCK, 10 * BLOCK])
    assert d["rows"][0]["n_samples_per_codebook"] == pytest.approx(BLOCK)
    assert d["rows"][1]["n_samples_per_codebook"] == pytest.approx(BLOCK)
    assert d["rows"][0]["n_frames"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 2. The MUI leg: fluctuation ~ sqrt(T/block), mean untouched
# --------------------------------------------------------------------------- #
def test_mui_fluctuation_scales_as_sqrt_t():
    """``rel.std = sqrt(2/K)/sqrt(F)`` with ``F = B/T`` gives ``sqrt(T)``."""
    s = [r["xi_rel_std"] for r in trade([1e4, 1e5, 1e6])["rows"]]
    assert s[1] / s[0] == pytest.approx(np.sqrt(10.0), rel=1e-6)
    assert s[2] / s[0] == pytest.approx(np.sqrt(100.0), rel=1e-6)


def test_mui_mean_is_independent_of_refresh_period():
    """Refreshing regularises MUI; it never reduces the mean.

    This is the most easily misread result in the project, so it is pinned
    directly: the mean ``(1/L) sum kappa`` must be identical for every T.
    """
    means = [r["xi_mean"] for r in trade_mui([1, 100, 1e6, 1e8])["rows"]]
    assert all(m == pytest.approx(means[0]) for m in means)
    assert means[0] == pytest.approx(mui.xi_block_mean(KAPPA_INT, L))
    assert trade([1])["n_interferers"] == M - 1


def test_frozen_baseline_reproduces_headline_fluctuation():
    """``T = block`` must give the known ``sqrt(2/K) = 53%`` for K = 7."""
    d = trade_mui([BLOCK])
    K = M - 1
    assert d["rows"][0]["xi_rel_std"] == pytest.approx(
        np.sqrt(2.0 / K), rel=1e-9)
    assert d["xi_rel_std_frozen"] == pytest.approx(0.5345, abs=1e-3)


def test_mui_eps_quantile_narrows_with_a_shorter_refresh_period():
    frozen, fast = trade_mui([BLOCK, 1e5])["rows"]
    mean = frozen["xi_mean"]
    assert frozen["xi_eps"] > mean
    assert fast["xi_eps"] > mean
    assert fast["xi_eps"] < frozen["xi_eps"]


# --------------------------------------------------------------------------- #
# 3. The PRF leg: cost ~ 1/T
# --------------------------------------------------------------------------- #
def test_prf_cost_is_inverse_in_t():
    a, b = trade([100, 1000])["rows"]
    assert a["prf_bits_per_symbol"] == pytest.approx(
        10.0 * b["prf_bits_per_symbol"])
    assert a["prf_bit_per_s"] == pytest.approx(10.0 * b["prf_bit_per_s"])


def test_prf_cost_formula():
    """``M*L/T`` bits per symbol, ``M*Rc/T`` bit/s."""
    r = trade([500])["rows"][0]
    assert r["prf_bits_per_symbol"] == pytest.approx(M * L / 500.0)
    assert r["prf_bit_per_s"] == pytest.approx(
        M * L / 500.0 * (CHIP_RATE / L))
    assert r["refresh_rate_hz"] == pytest.approx((CHIP_RATE / L) / 500.0)


# --------------------------------------------------------------------------- #
# 3b. mui.xi_frame_refresh must agree with attack.three_way_tradeoff
# --------------------------------------------------------------------------- #
# Why this block exists
# ---------------------
# ``mui.xi_frame_refresh`` was added later than ``three_way_tradeoff`` and had
# no test at all.  It reported ``refresh_rate_hz = 1 / n_symbols_per_frame`` --
# a rate per *symbol*, not per second -- so it disagreed with the (correct)
# ``three_way_tradeoff`` by a factor ``symbol_rate`` = 976 562.5.  The docs
# copied the wrong number and labelled it Hz.  The MUI statistics themselves
# were always right; only the rate was mislabelled.  These tests pin the two
# implementations together so the units cannot drift apart again.
def frame_refresh(T, **kw):
    return mui.xi_frame_refresh(KAPPA_INT, L, int(BLOCK) // int(T),
                                n_symbols_per_frame=int(T), n_codes=M,
                                symbol_rate=CHIP_RATE / L, **kw)


@pytest.mark.parametrize("T", [100.0, 1000.0, 10_000.0])
def test_frame_refresh_relative_std_matches_tradeoff(T):
    a = frame_refresh(T)["xi_rel_std"]
    b = trade_mui([T])["rows"][0]["xi_rel_std"]
    assert a == pytest.approx(b, rel=1e-12)


@pytest.mark.parametrize("T", [100.0, 1000.0, 10_000.0])
def test_frame_refresh_rate_matches_tradeoff(T):
    """Same refresh rate, in real Hz, from both entry points."""
    a = frame_refresh(T)["refresh_rate_hz"]
    b = trade_mui([T])["rows"][0]["refresh_rate_hz"]
    assert a == pytest.approx(b, rel=1e-12)
    # ... and the absolute value is the physical one, not 1/T in symbol units
    assert a == pytest.approx((CHIP_RATE / L) / T)


def test_frame_refresh_mean_is_independent_of_period():
    """Refreshing never changes the mean; only the fluctuation (§2.2)."""
    means = [frame_refresh(T)["xi_mean"] for T in (1, 100, 10 ** 6, BLOCK)]
    assert all(m == pytest.approx(means[0]) for m in means)
    assert means[0] == pytest.approx(mui.xi_block_mean(KAPPA_INT, L))


def test_frame_refresh_rejects_a_missing_time_base():
    """Without ``symbol_rate`` the rate must be 0.0, not a mislabelled 1/T."""
    d = mui.xi_frame_refresh(KAPPA_INT, L, 1000, n_symbols_per_frame=1000)
    assert d["refresh_rate_hz"] == 0.0
    assert d["prf_throughput_bits_per_s"] == 0.0
    # the statistical content is still available
    assert d["xi_rel_std"] > 0.0


def test_frame_refresh_fluctuation_grows_with_T():
    """The direction that is most often read backwards.

    ``T`` is *not* a sample count -- it is how long the code is held.  The
    number of independent codebooks averaged inside a block is ``F = block/T``,
    so a longer hold means **fewer** independent draws and **larger**
    fluctuation.  The sample count ``n`` is fixed at BLOCK throughout and does
    not appear in the formula.
    """
    stds = [frame_refresh(T)["xi_rel_std"]
            for T in (1, 100, 10_000, 10 ** 6, BLOCK)]
    assert all(a < b for a, b in zip(stds, stds[1:]))
    # frozen code (T = block) reproduces the headline sqrt(2/K)
    assert stds[-1] == pytest.approx(np.sqrt(2.0 / (M - 1)), rel=1e-12)


# --------------------------------------------------------------------------- #
# 4. The trade itself: the binding legs point opposite ways
# --------------------------------------------------------------------------- #
def test_security_and_prf_legs_conflict():
    """Shorter T is *better* for security and *worse* for PRF cost.

    This is the entire content of "three-way trade-off": there is no free
    direction.  MUI stability sides with security, which is why the feasible set
    is an interval rather than a point.
    """
    short, long_ = trade([100, 10_000])["rows"]
    assert short["margin"] < long_["margin"]                       # security
    assert short["xi_rel_std"] < long_["xi_rel_std"]               # MUI
    assert short["prf_bit_per_s"] > long_["prf_bit_per_s"]         # PRF


def test_eve_compute_dominates_bob_by_a_factor_of_l():
    """Per block Eve builds an ``L x L`` Gram matrix; Bob only despreads.

    ``Eve/Bob = L/M``: spreading is defensive on the compute axis even though it
    is *not* defensive on the statistical axis.
    """
    r = trade([1000])["rows"][0]
    assert r["eve_flops_per_block"] / r["bob_flops_per_block"] == pytest.approx(
        L / M)


# --------------------------------------------------------------------------- #
# 5. Consistency with the underlying closed forms
# --------------------------------------------------------------------------- #
def test_threshold_matches_subspace_identification_samples():
    d = trade([1.0])
    assert d["n_threshold"] == pytest.approx(
        attack.subspace_identification_samples(L, KAPPA))


def test_signal_ratio_rises_monotonically_to_a_positive_plateau():
    """``r`` increases with ``L`` and saturates at ``n_v/(M sigma^2 k)``.

    ``r = (L-M) n_v / [M (n_v + sigma^2 L k)]``.  Both limits are worth pinning
    because earlier versions of this project got them wrong:

    * ``r(M) = 0`` -- with as many chips as users the subspace is the whole
      space, so it carries no information;
    * ``r(inf) = n_v/(M sigma^2 k)``, a **positive** plateau, not zero.

    Consequence: extra chips stop helping once ``r`` reaches the plateau, which
    is a *negative* result for the "bigger L is safer" intuition.
    """
    rs = [attack.subspace_signal_ratio(Lv, KAPPA)["r_exact"]
          for Lv in (256, 512, 1024, 2048, 4096, 16384)]
    assert all(a < b for a, b in zip(rs, rs[1:]))       # rising
    plateau = attack.subspace_signal_ratio(L, KAPPA)["r_asymptotic"]
    assert plateau == pytest.approx(1.0 / M)            # n_v = s = k = 1
    assert all(r < plateau for r in rs)                 # never exceeds it
    assert rs[-1] > 0.999 * plateau                     # saturated by L=16384


def test_signal_ratio_vanishes_at_l_equals_m():
    """With exactly ``M`` chips the subspace fills the space: no information."""
    r = attack.subspace_signal_ratio(M, KAPPA)["r_exact"]
    assert r == pytest.approx(0.0)


def test_signal_fraction_falls_slightly_toward_the_plateau():
    """``1/(1+r)`` therefore *decreases* -- the opposite of the naive reading."""
    f = [attack.subspace_signal_ratio(Lv, KAPPA)["signal_fraction"]
         for Lv in (256, 1024, 4096, 16384)]
    assert all(a > b for a, b in zip(f, f[1:]))
    # plateau = M/(M+1) at n_v = sigma^2 = kappa = 1
    assert f[-1] == pytest.approx(M / (M + 1.0), abs=1e-2)


def test_signal_ratio_saturation_flag():
    assert not attack.subspace_signal_ratio(256, KAPPA)["saturated"]
    assert attack.subspace_signal_ratio(16384, KAPPA)["saturated"]


def test_recovery_probability_is_one_above_threshold():
    ok = attack.subspace_recovery_probability(L, KAPPA, n_samples=1e7)
    no = attack.subspace_recovery_probability(L, KAPPA, n_samples=10)
    assert ok["p_recover"] == 1.0 and ok["recovered"]
    assert no["p_recover"] == 0.0 and not no["recovered"]
    assert ok["margin"] > 1.0 > no["margin"]


# --------------------------------------------------------------------------- #
# 6. The performance regression that made E12 unusable
# --------------------------------------------------------------------------- #
def test_failed_attack_terminates_quickly():
    """A failed attack must not burn the iteration budget.

    The ``+-1`` sign iteration enters a short limit cycle whenever the subspace
    holds no ``+-1`` vector -- the situation for **every failed attack**.  An
    earlier cycle detector hashed a 1024-element tuple per iteration and made
    E12 effectively non-terminating; this pins termination rather than
    wall-clock, so it is machine independent.
    """
    rng = np.random.default_rng(5)
    Q, _ = np.linalg.qr(rng.standard_normal((256, 4)))   # generic subspace
    rec = attack.recover_codes_from_subspace(Q, iters=100_000, restarts=20,
                                             seed=1)
    assert rec["vectors"].shape[0] == 0
    assert rec["n_restarts"] == 20


def test_attack_succeeds_with_enough_samples_of_one_codebook():
    """Sanity check that the same helper does recover when n is large."""
    rng = np.random.default_rng(20261111)
    C = rng.choice([-1.0, 1.0], size=(4, 256))
    few = attack.code_recovery_from_covariance(
        C, np.ones(4), n_samples=100, restarts=200, seed=1)
    many = attack.code_recovery_from_covariance(
        C, np.ones(4), n_samples=20_000, restarts=200, seed=1)
    assert few["recovery_fraction"] == 0.0
    assert many["recovery_fraction"] >= 0.99
