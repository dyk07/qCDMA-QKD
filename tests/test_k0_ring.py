"""Bootstrap root ring: survive a failed bootstrap without an offline re-run.

Motivation
----------
The protocol needs a pre-shared root ``K_0`` to authenticate round 1, because the
round-1 code seed and auth key cannot come from a key that does not exist yet
(see ``docs/02_protocol.md`` §4).  Originally a failed round 1 meant an offline
re-bootstrap.  This module pins the cheap fix: hold several roots in advance and
**burn** the exposed one on failure.

The security-relevant points these tests lock:

1. Burn, never reuse.  A root that authenticated a tampered round may have been
   exposed, so it must never be used again; each root is derived from its own
   domain-separated label, so burning root ``i`` cannot help an attacker derive
   root ``i+1``.
2. Fail-closed is the default.  ``recover=False`` aborts, and the ring is left
   untouched.  Recovery is opt-in.
3. Recovery is bounded.  A ring tolerates *one-off* events; a persistent active
   attacker exhausts it and the loop still aborts.
4. Recovery only applies at the bootstrap.  A mid-loop failure is not a
   bootstrap problem and must never trigger a root burn.
5. The price is linear and tiny: one extra root per tolerated failure.
"""

from __future__ import annotations

import pytest

from qcdma_qkd import budget, loop
from qcdma_qkd.config import BudgetConfig, CDMAConfig, CVQKDConfig

CV = CVQKDConfig()
CDMA = CDMAConfig()
BD = BudgetConfig()
KW = dict(k0_bits=2 ** 20, cfg_cv=CV, cfg_cdma=CDMA, cfg_bud=BD,
          distance_km=25.0, n_symbols=10 ** 5,
          # these tests are about the *root ring*, not the size of the rate;
          # pin the analytic model so a finite-size block-length dependence
          # cannot abort round 1 (it does at n=1e5, 25 km in "finite" mode).
          rate_model="asymptotic")


def _run(n_rounds=5, **extra):
    return loop.run_loop(n_rounds, **KW, **extra)


# --------------------------------------------------------------------------- #
# the ring costs almost nothing
# --------------------------------------------------------------------------- #
def test_single_root_reproduces_the_old_behaviour():
    """n_k0_roots=1 must be byte-for-byte the pre-ring behaviour."""
    r = _run()
    assert r["aborted_at"] is None
    assert r["root_index"] == 0
    assert r["roots_burned"] == []
    assert r["roots_remaining"] == 0
    assert r["burned_attempts"] == []


def test_ring_budget_is_linear_in_root_count():
    one = budget.k0_ring_budget(BD, 10 ** 8, 0.4844, n_roots=1)
    three = budget.k0_ring_budget(BD, 10 ** 8, 0.4844, n_roots=3)
    assert three["total"] == pytest.approx(3 * one["total"])
    assert three["per_root"] == pytest.approx(one["per_root"])


def test_one_extra_root_is_the_price_of_one_failure():
    """Surviving one bootstrap failure costs exactly one more root."""
    one = budget.k0_ring_budget(BD, 10 ** 8, 0.4844, n_roots=1)
    two = budget.k0_ring_budget(BD, 10 ** 8, 0.4844, n_roots=2)
    assert two["cost_of_one_failure"] == pytest.approx(one["per_root"])
    # and that is ~282 bit, i.e. leaf-level noise next to a 4.3 Mbit round key
    assert 100 < one["per_root"] < 1000


def test_ring_size_must_be_positive():
    with pytest.raises(ValueError, match="n_k0_roots"):
        _run(n_k0_roots=0)


# --------------------------------------------------------------------------- #
# fail-closed is the default
# --------------------------------------------------------------------------- #
def test_failure_without_recovery_aborts_and_keeps_the_ring():
    r = _run(tamper_at=1, tamper_events=1, n_k0_roots=3, recover=False)
    assert r["aborted_at"] == 1
    assert r["roots_burned"] == [], "must not burn anything when not recovering"
    assert r["roots_remaining"] == 2


def test_recovery_is_opt_in():
    """Same inputs, only `recover` differs -- that must be the whole difference."""
    closed = _run(tamper_at=1, tamper_events=1, n_k0_roots=3, recover=False)
    open_ = _run(tamper_at=1, tamper_events=1, n_k0_roots=3, recover=True)
    assert closed["aborted_at"] == 1
    assert open_["aborted_at"] is None
    assert open_["roots_burned"] == [0]


# --------------------------------------------------------------------------- #
# recovery works on a one-off event
# --------------------------------------------------------------------------- #
def test_one_off_bootstrap_failure_recovers():
    r = _run(tamper_at=1, tamper_events=1, n_k0_roots=3, recover=True)
    assert r["aborted_at"] is None
    assert r["reason"] == "completed"
    assert r["root_index"] == 1
    assert r["roots_burned"] == [0]
    assert r["roots_remaining"] == 1


def test_recovered_run_has_a_clean_summary():
    """The burned attempt must not leak into `rounds` / the summary."""
    r = _run(tamper_at=1, tamper_events=1, n_k0_roots=3, recover=True)
    assert len(r["burned_attempts"]) == 1
    assert len(r["rounds"]) == 5
    assert all(rr.auth_ok for rr in r["rounds"])
    s = loop.loop_summary(r)
    assert s["auth_ok_all"] is True
    assert s["n_rounds"] == 5


def test_full_ring_tolerates_several_one_off_failures():
    """With 4 roots, three successive one-off failures are still survivable."""
    r = _run(tamper_at=1, tamper_events=3, n_k0_roots=4, recover=True)
    assert r["aborted_at"] is None
    assert r["roots_burned"] == [0, 1, 2]
    assert r["root_index"] == 3


# --------------------------------------------------------------------------- #
# recovery is bounded: a persistent attacker still wins
# --------------------------------------------------------------------------- #
def test_persistent_attack_exhausts_the_ring_and_aborts():
    r = _run(tamper_at=1, tamper_events=100, n_k0_roots=3, recover=True)
    assert r["aborted_at"] == 1
    assert r["roots_burned"] == [0, 1]
    assert r["roots_remaining"] == 0


def test_ring_of_one_cannot_recover():
    r = _run(tamper_at=1, tamper_events=1, n_k0_roots=1, recover=True)
    assert r["aborted_at"] == 1
    assert r["roots_burned"] == []


# --------------------------------------------------------------------------- #
# recovery must not apply outside the bootstrap
# --------------------------------------------------------------------------- #
def test_mid_loop_failure_never_burns_a_root():
    r = _run(n_rounds=6, tamper_at=3, tamper_events=1, n_k0_roots=5,
             recover=True)
    assert r["aborted_at"] == 3
    assert r["roots_burned"] == [], (
        "a mid-loop failure is not a bootstrap problem; burning a root would "
        "not help and would hide the real fault")


# --------------------------------------------------------------------------- #
# roots are independent (burn one -> the next is unaffected)
# --------------------------------------------------------------------------- #
def test_burned_root_does_not_determine_the_next():
    """Each root is derived from its own label, never from its predecessor."""
    a = loop.kdf_expand(loop.hashlib.sha256(b"K0-root/0").digest(),
                        loop.TAG_AUTH, 32)
    b = loop.kdf_expand(loop.hashlib.sha256(b"K0-root/1").digest(),
                        loop.TAG_AUTH, 32)
    assert a != b
    # learning root 0 must not let anyone recompute root 1: the labels are a
    # plain counter, so independence is exactly the one-wayness of the root
    # derivation (sha256), not of the KDF expansion.
    assert (loop.hashlib.sha256(b"K0-root/0").digest()
            != loop.hashlib.sha256(b"K0-root/1").digest())


def test_recovery_restarts_at_round_one_under_the_new_root():
    r = _run(tamper_at=1, tamper_events=1, n_k0_roots=2, recover=True)
    assert r["rounds"][0].round_index == 1
    assert r["burned_attempts"][0].round_index == 1
    # the retried round 1 must succeed
    assert r["rounds"][0].auth_ok
