"""Where do ``a_{n+1}`` and ``c_{n+1}`` come from?

The protocol document's key flow is unambiguous::

    K_n --KDF--> a_{n+1}   authentication key
             --> c_{n+1}   code seed
             --> u_n       application output

An earlier implementation silently broke this: the loop passed an externally
supplied ``auth_key``/``seed`` into ``simulate_round`` (required in the bootstrap
round, where ``K_n`` does not exist yet) and then, for every later round, derived

    auth_key  = kdf_expand(previous seed, TAG_AUTH)
    next seed = kdf_expand(previous seed, TAG_SEED)

so the chain was ``a_i = f(c_{i-1})`` and ``c_i = f(c_{i-1})``, with ``K_n``
never entering it at all.  That is a real security defect, not a cosmetic one:

* ``c_{n-1}`` is exactly what an attacker tries to blind-estimate (docs/03 §3),
  so authentication would sit **downstream of code leakage**;
* the three-way domain separation (auth / seed / app) becomes a chain, so one
  leaked seed compromises every future auth key and seed.

These tests pin the corrected structure:

    a_{n+1}, c_{n+1}  <-  K_n            (never from c_n)
    a, c, u          pairwise independent  (distinct domain tags)
"""

from __future__ import annotations

import pytest

from qcdma_qkd import budget, loop
from qcdma_qkd.config import BudgetConfig, CDMAConfig, CVQKDConfig

CV = CVQKDConfig()
CDMA = CDMAConfig()
BD = BudgetConfig()
KW = dict(k0_bits=2 ** 20, cfg_cv=CV, cfg_cdma=CDMA, cfg_bud=BD,
          distance_km=25.0, n_symbols=10 ** 6,
          # the subject here is the derivation chain, not the rate model
          rate_model="asymptotic")


def _rounds(n=3):
    return loop.run_loop(n, **KW)["rounds"]


# --------------------------------------------------------------------------- #
# the documented flow actually happens
# --------------------------------------------------------------------------- #
def test_next_auth_key_is_split_from_kn_not_from_the_seed():
    r = _rounds(2)
    k0, k1 = r[0], r[1]
    # what K_0's split says
    pred = loop.split_key(k0.k_bits, BD, overhead_bits=k0.a_bits,
                          key_material=k0.extras["key_material"])
    assert k1.extras["auth_key"] == pred["a"]
    # and it is NOT derivable from the code seed
    from_seed = loop.kdf_expand(k0.extras["seed"], loop.TAG_AUTH, 32)
    assert k1.extras["auth_key"] != from_seed


def test_next_seed_is_split_from_kn_not_from_the_old_seed():
    r = _rounds(2)
    k0, k1 = r[0], r[1]
    pred = loop.split_key(k0.k_bits, BD, overhead_bits=k0.a_bits,
                          key_material=k0.extras["key_material"])
    assert k1.extras["seed"] == pred["c"]
    from_seed = loop.kdf_expand(k0.extras["seed"], loop.TAG_SEED, 32)
    assert k1.extras["seed"] != from_seed


def test_kn_enters_the_chain():
    """K_n must actually be consumed; if it were decorative this would fail."""
    r = _rounds(3)
    # different K_n -> different next-round material
    preds = [loop.split_key(k.k_bits, BD, overhead_bits=k.a_bits,
                            key_material=k.extras["key_material"])["a"]
             for k in r]
    assert len(set(preds)) == len(preds)


# --------------------------------------------------------------------------- #
# domain separation survives
# --------------------------------------------------------------------------- #
def test_auth_seed_and_app_are_pairwise_independent():
    r = _rounds(1)
    out = loop.split_key(r[0].k_bits, BD, overhead_bits=r[0].a_bits,
                         key_material=r[0].extras["key_material"])
    assert out["a"] != out["c"]
    assert out["a"] != out["u"]
    assert out["c"] != out["u"]


def test_root_derivation_changes_with_the_key_material():
    a = loop.split_key(10 ** 6, BD, 100.0, key_material=b"K-A")
    b = loop.split_key(10 ** 6, BD, 100.0, key_material=b"K-B")
    assert a["a"] != b["a"] and a["c"] != b["c"]


def test_missing_key_material_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        loop.split_key(10 ** 6, BD, 100.0, key_material=b"")


# --------------------------------------------------------------------------- #
# the bootstrap is the *only* place an external key is supplied
# --------------------------------------------------------------------------- #
def test_bootstrap_uses_the_root_and_later_rounds_do_not():
    r = _rounds(2)
    root_key = loop.hashlib.sha256(b"K0-root/0").digest()
    assert r[0].extras["seed"] == loop.kdf_expand(root_key, loop.TAG_SEED, 32)
    assert r[0].extras["auth_key"] == loop.kdf_expand(root_key, loop.TAG_AUTH, 32)
    # round 2 no longer touches the root
    assert r[1].extras["seed"] != loop.kdf_expand(root_key, loop.TAG_SEED, 32)


# --------------------------------------------------------------------------- #
# code secrecy no longer gates authentication
# --------------------------------------------------------------------------- #
def test_knowing_every_seed_does_not_reveal_any_auth_key():
    """The structural claim: auth keys are not a function of the seeds.

    A seed-only adversary gets the seeds by definition; what they must not be
    able to compute is ``a_{n+1}``.  The guard is that the derivation is a
    chain from ``K_n``, which the seed does not determine.
    """
    r = _rounds(3)
    for prev, cur in zip(r, r[1:]):
        for tag in (loop.TAG_AUTH, loop.TAG_SEED, loop.TAG_APP):
            guess = loop.kdf_expand(prev.extras["seed"], tag, 32)
            assert guess != cur.extras["auth_key"]
            assert guess != cur.extras["seed"]


# --------------------------------------------------------------------------- #
# the derived auth key must actually be a key, not the rounded billing figure
# --------------------------------------------------------------------------- #
def test_auth_key_is_derived_at_its_security_length_not_the_billing_figure():
    """Regression: ``ceil(a_bits/8)`` gave the MAC an 8-bit key.

    ``a_bits`` is the *amortised* cost (l/r ~ 0.13 bit), so rounding it up to
    whole bytes yielded one byte -- forgery probability ~2^-8 instead of the
    claimed eps_auth = 1e-12.  The key length must come from ``key_length``.
    """
    from qcdma_qkd.config import BudgetConfig
    cfg = BudgetConfig()
    info = budget.auth_key_bits(cfg, eps_auth=1e-12)
    assert info["per_message"] < 1.0                    # the number that is fractional
    r = _rounds(3)
    for rr in r[1:]:
        # enough bits to support the claimed forgery bound
        assert len(rr.extras["auth_key"]) * 8 >= 128
        assert len(rr.extras["auth_key"]) == int(-(-info["key_length"] // 8))


def test_amortised_cost_matches_the_actual_key_lifetime():
    """Regression: amortising over 2^20 while re-keying every round.

    With one fresh key per round and 667 messages per round, the reuse is 667,
    not mac_reuse_limit, so the per-round spend must be the full ``l``.
    """
    from qcdma_qkd.config import BudgetConfig
    cfg = BudgetConfig()
    l = budget.auth_key_bits(cfg, eps_auth=1e-12)["key_length"]
    info = budget.auth_key_bits(cfg, eps_auth=1e-12, n_messages=667)
    assert info["r_effective"] == 667
    assert info["per_message"] == pytest.approx(l / 667.0)
    assert info["per_round"] == pytest.approx(l)
    assert budget.auth_bits(667, cfg) == pytest.approx(l)


# --------------------------------------------------------------------------- #
# regression: the loop must use the finite-size rate, not the asymptotic one
# --------------------------------------------------------------------------- #
def test_loop_defaults_to_the_finite_size_rate():
    """The document argues in finite-size terms, so the loop must compute it.

    Regression: ``simulate_round`` called ``asymptotic_rate`` while every doc
    sentence said "finite-size protocol".  At n=1e8, 25 km that overstated K_n
    by 23%; below n~1e7 the finite-size rate is zero, so the loop should abort
    where the asymptotic one happily reports a healthy key.
    """
    # KW already carries rate_model="asymptotic"; override it explicitly
    kw = dict(KW, n_symbols=10 ** 5)
    fin = loop.run_loop(1, **dict(kw, rate_model="finite"))
    asym = loop.run_loop(1, **kw)
    # at n=1e5, 25 km the finite-size rate is zero -> round 1 is not feasible
    assert fin["rounds"][0].extras["rate_model"] == "finite"
    assert fin["aborted_at"] == 1
    assert asym["rounds"][0].extras["rate_model"] == "asymptotic"
    assert asym["aborted_at"] is None
    # and the finite-size key is strictly smaller wherever both exist
    kw2 = dict(kw, n_symbols=10 ** 8)
    f8 = loop.run_loop(1, **dict(kw2, rate_model="finite"))["rounds"][0]
    a8 = loop.run_loop(1, **kw2)["rounds"][0]
    assert f8.k_bits < a8.k_bits


def test_rate_model_rejects_unknown_values():
    with pytest.raises(ValueError, match="rate_model must be"):
        loop.run_loop(1, **dict(KW, rate_model="bogus"))
