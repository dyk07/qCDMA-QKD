"""Frame-level code refresh: the three-way trade-off.

Why this file exists
--------------------
Two project results pull in opposite directions and must be reconciled in one
place:

* ``mui.xi_block_std`` -- the block-averaged excess noise has relative
  fluctuation ``sqrt(2/K)`` (53% at ``K = 7``) and it does **not** shrink with
  block length, because the codes are frozen for the whole block.
* ``attack.code_recovery_from_covariance`` -- the same freezing is what lets Eve
  estimate one ``Sigma`` from many symbols and read the codes out of it.

Both are consequences of "the code repeats".  Refreshing the codebook every
``T`` symbols moves along a single axis and trades three things:

    security  (larger T -> more samples per codebook -> easier attack)
    MUI       (smaller T -> more frames -> smaller fluctuation)
    PRF cost  (smaller T -> more refreshes -> more PRF throughput)

``mui.xi_frame_refresh`` quantifies the MUI leg; this file pins all three.

The honest limit
----------------
"Attack blocked at small T" is a statement about the **second-moment** attack
only.  It is not a cryptographic security claim: higher moments, the PRF
structure, and correlated-symbol attacks are unmodelled.
"""

from __future__ import annotations

import numpy as np
import pytest

from qcdma_qkd import attack, mui

L = 1024
K = 7                       # interferers in the default 8-user scenario
KAPPA = np.ones(K)


# --------------------------------------------------------------------------- #
# 1. The MUI leg: mean unchanged, fluctuation shrinks as 1/sqrt(F)
# --------------------------------------------------------------------------- #
def test_frame_refresh_preserves_the_mean():
    """A fresh code is just as correlated as an old one *on average*.

    Refreshing is not a way to reduce MUI -- it only regularises it.  If this
    ever changed the mean, the whole E2/E9 analysis would be invalidated.
    """
    base = mui.xi_frame_refresh(KAPPA, L, n_frames=1)
    for F in (2, 10, 1000):
        r = mui.xi_frame_refresh(KAPPA, L, n_frames=F)
        assert r["xi_mean"] == pytest.approx(base["xi_mean"])
    assert base["xi_mean"] == pytest.approx(mui.xi_block_mean(KAPPA, L))


def test_frame_refresh_std_scales_as_inverse_sqrt_frames():
    """``Var ~ 1/F``: the fluctuation is the thing that averages down."""
    base = mui.xi_frame_refresh(KAPPA, L, n_frames=1)
    for F in (4, 25, 400):
        r = mui.xi_frame_refresh(KAPPA, L, n_frames=F)
        assert r["xi_std"] == pytest.approx(base["xi_std"] / np.sqrt(F), rel=1e-9)
        assert r["xi_rel_std"] == pytest.approx(
            base["xi_rel_std"] / np.sqrt(F), rel=1e-9)


def test_frozen_baseline_reproduces_the_known_fluctuation():
    """``F = 1`` must reproduce ``sqrt(2/K) = 53%`` -- the project's headline."""
    r = mui.xi_frame_refresh(KAPPA, L, n_frames=1)
    assert r["xi_rel_std"] == pytest.approx(np.sqrt(2.0 / K), rel=1e-9)
    assert r["xi_rel_std"] == pytest.approx(0.5345, abs=1e-3)


def test_frame_refresh_eps_quantile_narrows_with_frames():
    """Provisioning at eps=1e-12 costs less when the code is refreshed."""
    q1 = mui.xi_frame_refresh(KAPPA, L, n_frames=1)["xi_eps"]
    q100 = mui.xi_frame_refresh(KAPPA, L, n_frames=100)["xi_eps"]
    mean = mui.xi_block_mean(KAPPA, L)
    assert q1 > mean and q100 > mean
    assert q100 < q1, "refreshing must reduce the tail, not increase it"
    # the overshoot (relative to the mean) should shrink roughly as 1/sqrt(F)
    r1 = (q1 - mean) / mean
    r100 = (q100 - mean) / mean
    assert r100 < r1 / 5.0


def test_frame_refresh_unequal_powers():
    """The Gamma route must be monotone too (no closed chi^2 form)."""
    kappa = np.linspace(1.0, 0.3, K)
    a = mui.xi_frame_refresh(kappa, L, n_frames=1)
    b = mui.xi_frame_refresh(kappa, L, n_frames=50)
    assert a["xi_mean"] == pytest.approx(b["xi_mean"])
    assert b["xi_std"] < a["xi_std"]
    assert b["xi_eps"] < a["xi_eps"]


# --------------------------------------------------------------------------- #
# 2. The PRF leg: cost is inversely proportional to T
# --------------------------------------------------------------------------- #
def test_prf_cost_is_inverse_in_refresh_period():
    """``M * L`` bits per refresh, ``1/T`` refreshes per symbol.

    ``M`` is the number of codes in the codebook (victim + interferers = 8),
    **not** ``kappa.size`` (7 interferers).  That distinction is exactly why
    ``xi_frame_refresh`` takes an explicit ``n_codes``.
    """
    a = mui.xi_frame_refresh(KAPPA, L, n_frames=10, n_symbols_per_frame=100,
                             n_codes=K + 1)
    b = mui.xi_frame_refresh(KAPPA, L, n_frames=10, n_symbols_per_frame=200,
                             n_codes=K + 1)
    assert a["prf_throughput_bits_per_s"] == pytest.approx(
        2.0 * b["prf_throughput_bits_per_s"])
    assert a["refresh_rate_hz"] == pytest.approx(2.0 * b["refresh_rate_hz"])
    assert a["prf_throughput_bits_per_s"] == pytest.approx(
        (K + 1) * L * 8 / 100.0)


@pytest.mark.parametrize("n_codes,expected", [(7, 573.44), (8, 655.36)])
def test_prf_cost_uses_the_declared_code_count(n_codes, expected):
    """The default counts one more code than ``kappa`` holds (the victim)."""
    r = mui.xi_frame_refresh(KAPPA, L, n_frames=1, n_symbols_per_frame=100,
                             n_codes=n_codes)
    assert r["prf_throughput_bits_per_s"] == pytest.approx(expected)
    assert r["n_codes"] == n_codes


def test_prf_cost_default_includes_the_victim():
    r = mui.xi_frame_refresh(KAPPA, L, n_frames=1, n_symbols_per_frame=100)
    assert r["n_codes"] == K + 1


# --------------------------------------------------------------------------- #
# 3. The security leg: fewer samples per codebook must hurt the attack
# --------------------------------------------------------------------------- #
def test_attack_on_a_single_codebook_needs_a_minimum_number_of_symbols():
    """Recovery is monotone in the samples available for one ``Sigma``.

    This is the security leg of the trade-off: it is the same measurement as
    E11, viewed as a function of the refresh period.
    """
    rng = np.random.default_rng(20261110)
    C = rng.choice([-1.0, 1.0], size=(4, 256))

    few = attack.code_recovery_from_covariance(
        C, np.ones(4), n_samples=100, restarts=200, iters=2000, seed=1)
    many = attack.code_recovery_from_covariance(
        C, np.ones(4), n_samples=20_000, restarts=200, iters=2000, seed=1)

    assert few["recovery_fraction"] == 0.0
    assert many["recovery_fraction"] >= 0.99
    # the operator-norm error is what separates the two regimes
    assert few["err_operator_norm"] > 3.0 * many["err_operator_norm"]


def test_cycle_detection_terminates_quickly_on_a_failed_attack():
    """A failed attack must not burn the full iteration budget.

    The sign iteration falls into a short limit cycle when the subspace holds
    no ``+-1`` vector, which is the common case *during a failed attack*.  The
    cycle guard is what keeps E12's scan tractable, so it is worth pinning.
    """
    rng = np.random.default_rng(5)
    Q, _ = np.linalg.qr(rng.standard_normal((256, 4)))   # generic subspace
    rec = attack.recover_codes_from_subspace(Q, iters=100_000, restarts=20,
                                             seed=1)
    # a generic subspace contains no +-1 vector
    assert rec["vectors"].shape[0] == 0
    assert rec["n_restarts"] == 20
