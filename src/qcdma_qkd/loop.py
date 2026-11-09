"""Closed-loop q-CDMA protocol simulation.

This module implements the *classical plumbing* of the protocol described in
``docs/02_protocol.md`` and checks it end-to-end:

  round 0 : pre-shared authentication root K_0
  round n : QKD block -> K_n -> KDF -> (a_n, c_n, u_n)
            codebook    = PRF_{c_n}(frame, user, session) expanded locally
            control     = MAC_{a_n}(code descriptor, sync, PE commitments)
            quantum     = coherent states with the derived codes
            post-proc   = PE / reconciliation / privacy amplification -> K_{n+1}

The KDF is implemented with a keyed BLAKE2b in counter mode, which is a
*demonstration* of the three-way split (domain-separated, independent outputs),
**not** the information-theoretic extractor the security proof assumes.  The gap
is stated explicitly in ``docs/05_security.md``.

Code model (the project's final answer for the sample scenario)
---------------------------------------------------------------
``CDMAConfig.code_model`` defaults to ``"prf"`` with ``refresh_fraction = 1.0``:

* a PRF expands the seed into the codebook, so observing chips does not leak
  seed bits -- there is **no** refresh constraint, and
* ``f = 1`` gives a fully fresh codebook every round, which kills cross-round
  correlation outright.

The price is that the code layer's secrecy is **computational** (a PRF
assumption), not information-theoretic.  The information-theoretic route
(``code_model = "extractor"``) is fully implemented: it needs ``f >= f_min``
because a 256-bit seed cannot absorb ``L = 1024`` known chips -- see
``CDMAConfig.refresh_constraint()`` and ``docs/05_security.md`` §5.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field

import numpy as np

from . import budget, codes, cvqkd, mui, security
from .config import BudgetConfig, CDMAConfig, CVQKDConfig, SecurityConfig

# domain-separation tags -- never reuse a key across a different purpose
TAG_AUTH = b"qcdma/v1/auth"
TAG_SEED = b"qcdma/v1/seed"
TAG_APP = b"qcdma/v1/app"


# --------------------------------------------------------------------------- #
# KDF / PRF primitives
# --------------------------------------------------------------------------- #
def kdf_expand(key: bytes, tag: bytes, length_bytes: int) -> bytes:
    """Keyed BLAKE2b in counter mode with an explicit domain tag."""
    out = bytearray()
    counter = 0
    while len(out) < length_bytes:
        h = hashlib.blake2b(digest_size=64, key=key[:64])
        h.update(tag)
        h.update(counter.to_bytes(8, "big"))
        out += h.digest()
        counter += 1
    return bytes(out[:length_bytes])


def split_key(k_bits: float, cfg: BudgetConfig, overhead_bits: float = 0.0,
              key_material: bytes | None = None) -> dict:
    """Three-way KDF split with independent domain tags.

    This is the **production** derivation used by the loop: round ``n``'s key
    ``K_n`` is split into the *next* round's auth key, the next round's code
    seed, and this round's application output::

        K_n -> a_{n+1}, c_{n+1}, u_n          (independent domain tags)

    ``overhead_bits`` is the authentication cost paid out of the next round's
    key (typically ``budget.round_auth_bits``).

    Parameters
    ----------
    key_material
        The actual secret ``K_n``.  When given, the three streams are derived
        from it, so they are secret **as long as K_n is** -- which CV-QKD
        guarantees and which has nothing to do with code secrecy.  When
        omitted (test / sizing use only) a placeholder root is used; that
        variant carries **no** security claim and must never be shipped into a
        real derivation chain.
    """
    a_bits = float(overhead_bits)
    c_bits = float(cfg.seed_bits)
    u_bits = float(k_bits) - a_bits - c_bits
    if u_bits < 0:
        raise ValueError(f"key exhausted: K={k_bits} < a+c={a_bits + c_bits}")
    if key_material is not None:
        if not key_material:
            raise ValueError("key_material must be non-empty")
        root = hmac.new(key_material, b"qcdma/v1/root", hashlib.sha256).digest()
    else:
        # sizing only -- NOT a secret; see the docstring warning above
        root = hashlib.sha256(f"K-{k_bits}".encode()).digest()

    # The auth key must be a real key, not the rounded *billing* figure.  The
    # cost ``a_bits`` is amortised (l/r, a fraction of a bit) while the key
    # itself is ``l = t + 2log2(1/eps)``.  Deriving ceil(a_bits/8) bytes would
    # hand the MAC an 8-bit key and silently destroy eps_auth, so the length is
    # driven by the key requirement instead.
    kereq = budget.auth_key_bits(cfg, n_messages=1)["key_length"]

    def _bytes(bits: float) -> int:
        return int(np.ceil(max(bits, 1) / 8.0))

    return dict(
        a_bits=a_bits, c_bits=c_bits, u_bits=u_bits,
        a=kdf_expand(root, TAG_AUTH, _bytes(kereq)),
        c=kdf_expand(root, TAG_SEED, _bytes(c_bits)),
        u=kdf_expand(root, TAG_APP, min(_bytes(u_bits), 1 << 20)),
    )


def derive_codebook(seed: bytes, n_users: int, L: int, frame: int,
                    session: int = 0,
                    base: np.ndarray | None = None,
                    refresh_fraction: float = 1.0,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """PRF-based codebook: chips are the bits of a keyed BLAKE2b stream.

    ``c_i[l] = 2 * bit_l( PRF(seed, i, frame, session) ) - 1``

    With ``base`` given, only a ``refresh_fraction`` fraction of chips is
    re-drawn from the new PRF stream and the rest is carried over from
    ``base``; this is what makes the refresh constraint of
    ``docs/05_security.md`` §5 binding.  With ``refresh_fraction = 1`` (the
    project default) ``base`` is irrelevant and every chip is fresh.

    Security note (regression)
    --------------------------
    The description *and* the selection of the refreshed chips must both come
    from the **secret** PRF stream.  An earlier version drew the fresh chips
    (and the refresh mask) from ``np.random.default_rng(frame)``, i.e. from a
    public, frame-indexed source that is independent of ``seed``.  The codebook
    then did not depend on the seed at all for ``f < 1``, so refreshing bought
    neither secrecy nor decorrelation.  Both are now taken from the PRF stream.
    """
    n_bytes = int(np.ceil(L / 8.0))
    rows = np.empty((n_users, L), dtype=np.float64)
    for i in range(n_users):
        stream = kdf_expand(seed, TAG_SEED + f"/u{i}/f{frame}/s{session}".encode(),
                            n_bytes)
        bits = np.unpackbits(np.frombuffer(stream, dtype=np.uint8))[:L]
        rows[i] = 2.0 * bits - 1.0
    if base is not None and float(refresh_fraction) < 1.0:
        # Refresh mask from the PRF stream too -- never from a public RNG.
        # A 32-bit uniform word per chip is needed: single bits can only
        # express a refresh probability of 0 or 1/2, which would silently
        # mis-scale f (f=0.75 would refresh 50% of the chips).
        mask_bytes = kdf_expand(
            seed, TAG_SEED + f"/mask/f{frame}/s{session}".encode(), 4 * L)
        words = np.frombuffer(mask_bytes, dtype=">u4")
        fresh = words < np.uint32(int(round(float(refresh_fraction) * 2 ** 32))
                                  if float(refresh_fraction) < 1.0 else 2 ** 32 - 1)
        out = np.array(base, dtype=np.float64, copy=True)
        out[:, fresh] = rows[:, fresh]
        return out
    return rows


def mac(key: bytes, message: bytes, tag_bytes: int = 16) -> bytes:
    """Truncated HMAC-SHA256 (stand-in for an eps-AXU family in the demo)."""
    return hmac.new(key, message, hashlib.sha256).digest()[:tag_bytes]


def verify_mac(key: bytes, message: bytes, tag: bytes) -> bool:
    return hmac.compare_digest(mac(key, message, len(tag)), tag)


# --------------------------------------------------------------------------- #
# round result
# --------------------------------------------------------------------------- #
@dataclass
class RoundResult:
    round_index: int
    k_bits: float                 # raw key produced by the QKD block
    a_bits: float
    c_bits: float
    u_bits: float
    auth_ok: bool
    xi_total: float
    xi_mui: float
    rate_per_symbol: float
    feasible: bool
    codebook: np.ndarray | None = None
    extras: dict = field(default_factory=dict)


def simulate_round(round_index: int, seed: bytes, n_symbols: int,
                   cfg_cv: CVQKDConfig, cfg_cdma: CDMAConfig,
                   cfg_bud: BudgetConfig, distance_km: float,
                   interferer_powers: np.ndarray | None = None,
                   eps_mui: float = 1e-12,
                   tamper: bool = False,
                   auth_key: bytes | None = None,
                   prev_codebook: np.ndarray | None = None,
                   rate_model: str = "finite",
                   eps_sm: float = 1e-10,
                   eps_pa: float = 1e-12) -> RoundResult:
    """One closed-loop round with the real budget and MUI accounting.

    ``K_n`` is *computed*, not supplied: it is the raw post-PA key produced by
    the quantum block, ``rate_per_symbol * n_symbols``.  The auth key is passed
    in explicitly because it comes from the *previous* round.

    ``rate_model`` selects the quantum-layer rate:

    ``"finite"`` (default)
        :func:`cvqkd.finite_size_rate` -- the AEP + PE penalty for the actual
        block length, with the PE sample removed.  This is the model the rest
        of the document argues in ("finite-size protocol"), so it must be the
        one the loop uses.  An earlier version used the asymptotic rate here
        while every doc sentence talked about finite-size, which overstated the
        closed-loop rate by 23% at ``n = 1e8`` and by 100% below ``n ~ 1e7``.

    ``"asymptotic"``
        Kept for the analytic comparisons in ``docs/03`` and for regression
        tests that pin the closed form.  Never use it for headline numbers.
    """
    L = cfg_cdma.spreading_factor
    # --- codebook -------------------------------------------------------- #
    # ``prev_codebook`` carries the previous round's codes so that the
    # configured ``refresh_fraction`` is actually honoured (it used to be a
    # free parameter that nothing consumed).  For the project default
    # (code_model='prf', f=1) every chip is fresh and the constraint of
    # docs/05 §5 does not bind.
    codebook = derive_codebook(seed, cfg_cdma.n_users, L, round_index,
                               base=prev_codebook,
                               refresh_fraction=cfg_cdma.refresh_fraction)
    # --- MUI -> excess noise (worst-case quantile, NOT the mean) ---------- #
    kappa = (mui.uniform_kappa(cfg_cdma.n_users - 1, 1.0)
             if interferer_powers is None
             else np.asarray(interferer_powers, dtype=float))
    xi_mui = mui.required_xi(eps_mui, kappa, L)
    # --- quantum layer ---------------------------------------------------- #
    T = cvqkd.total_transmittance(distance_km, cfg_cv.alpha_db_per_km, cfg_cv.eta)
    xi_tot = cfg_cv.excess_noise + xi_mui
    if rate_model == "asymptotic":
        r = cvqkd.asymptotic_rate(cfg_cv.V_A, T, xi_tot, cfg_cv.eta, cfg_cv.v_el,
                                  cfg_cv.beta, cfg_cv.detection)
    elif rate_model == "finite":
        r = cvqkd.finite_size_rate(
            cfg_cv.V_A, T, xi_tot, cfg_cv.eta, cfg_cv.v_el, cfg_cv.beta,
            cfg_cv.detection, n_symbols=int(n_symbols), eps_sm=eps_sm,
            eps_pa=eps_pa, sample_fraction=cfg_bud.pe_sample_fraction)
    else:
        raise ValueError("rate_model must be 'finite' or 'asymptotic', got %r"
                         % (rate_model,))
    i_ab = r["I_AB"]
    # ``finite_size_rate`` already reports S per *total* symbol (the PE sample
    # is inside n_symbols), so K_n = S * n_symbols in both models.
    k_bits = r["S_pos"] * n_symbols
    # --- key budget ------------------------------------------------------- #
    b = budget.round_budget(k_bits, n_symbols, i_ab, cfg_bud)
    # --- authentication of THIS round's control channel ------------------- #
    # (a_n from the previous round; K_0 in the bootstrap round)
    auth_key = auth_key if auth_key is not None else kdf_expand(seed, TAG_AUTH, 32)
    descriptor = b"code-update/" + bytes([round_index % 256]) + seed[:8]
    tag = mac(auth_key, descriptor)
    received = bytearray(descriptor)
    if tamper:
        received[-1] ^= 0xFF
    auth_ok = verify_mac(auth_key, bytes(received), tag)
    # Secret material for K_n.  In a real implementation this is the raw post-PA
    # key produced by the quantum block.  Here it is derived deterministically
    # from the round parameters so runs are reproducible; crucially it does NOT
    # depend on the code seed, so the next round's auth key and seed stay
    # independent of code secrecy.
    key_material = hashlib.sha256(
        b"Kn/%d/%d/%.9f/%.9f" % (round_index, n_symbols, xi_tot, T)).digest()
    return RoundResult(round_index=round_index, k_bits=float(k_bits),
                       a_bits=b["a"], c_bits=b["c"], u_bits=b["u"],
                       auth_ok=auth_ok, xi_total=xi_tot, xi_mui=xi_mui,
                       rate_per_symbol=r["S_pos"], feasible=b["feasible"],
                       codebook=codebook,
                       extras=dict(I_AB=i_ab, chi_BE=r["chi_BE"],
                                   n_messages=b["n_messages"],
                                   surplus=b["surplus"], T=T,
                                   seed=seed, auth_key=auth_key,
                                   key_material=key_material,
                                   code_model=cfg_cdma.code_model,
                                   refresh_fraction=cfg_cdma.refresh_fraction,
                                   rate_model=rate_model))


# --------------------------------------------------------------------------- #
# multi-round loop
# --------------------------------------------------------------------------- #
def run_loop(n_rounds: int, k0_bits: int, cfg_cv: CVQKDConfig,
             cfg_cdma: CDMAConfig, cfg_bud: BudgetConfig,
             distance_km: float, n_symbols: int = 10 ** 8,
             eps_mui: float = 1e-12,
             interferer_powers: np.ndarray | None = None,
             tamper_at: int | None = None,
             tamper_events: int = 1,
             n_k0_roots: int = 1,
             recover: bool = False,
             eps_pe: float = 1e-10,
             pe_reject_at: int | None = None,
             near_miss_ratio: float = 1e-3,
             enforce_budget: bool = True,
             rate_model: str = "finite",
             eps_sm: float = 1e-10,
             eps_pa: float = 1e-12) -> dict:
    """Run the closed loop, bootstrapping from a ring of pre-shared roots.

    Accounting model (the point of the whole exercise):

    * round 1 authenticates with root 0; thereafter with ``a_{n-1}``, a
      sub-key of round n-1's key.  The *root* is never reused.
    * ``K_n`` is regenerated from scratch by the quantum block, so the loop does
      **not** consume net output in steady state -- it consumes only the
      ``a_n + c_n`` slice of each round's own key.

    Declared failure modes (now actually enforced)
    ----------------------------------------------
    The protocol document lists four abort conditions.  Three of them used to be
    prose only; they are checked here so that a run cannot silently continue on
    an infeasible design:

    1. **Authentication failure** -> abort (the fail-closed default).
    2. **Budget infeasible** (``u_n < 0``, i.e. the round key cannot cover
       auth + seed) -> abort with reason ``"budget infeasible"``.
       Suppressed by ``enforce_budget=False`` (useful only to *study* the
       infeasible region).
    3. **Parameter estimation rejected** (estimated excess noise exceeds the
       design value by more than the ``eps_pe`` confidence bound) -> abort with
       reason ``"parameter estimation rejected"``.  Modelled by
       ``pe_reject_at``.
    4. **Root exhausted** (no root left for the bootstrap) -> abort with reason
       ``"K_0 exhausted"``.

    Rounds that *survive* but sit within a factor ``near_miss_ratio`` of the
    infeasibility boundary are recorded in ``near_misses``; a run that spends
    most of its time there is not numerically wrong but is operationally
    fragile, and that should be visible rather than inferred from a thin
    surplus.
    """
    if n_k0_roots < 1:
        raise ValueError("n_k0_roots must be >= 1")

    def _root_key(i: int) -> bytes:
        # distinct, domain-separated root per index -- never derive two roots
        # from one another, so burning one cannot compromise the next
        return hashlib.sha256(b"K0-root/%d" % i).digest()

    rounds: list[RoundResult] = []
    burned_attempts: list[RoundResult] = []
    near_misses: list[int] = []
    k0_left = float(k0_bits)
    keys: list[float] = []
    root_index = 0
    roots_burned: list[int] = []
    tamper_budget = int(tamper_events)

    k0_key = _root_key(root_index)
    auth_key = kdf_expand(k0_key, TAG_AUTH, 32)
    seed_bytes = kdf_expand(k0_key, TAG_SEED, 32)
    prev_codebook: np.ndarray | None = None

    def _finish(reason: str, at: int | None) -> dict:
        return dict(rounds=rounds, aborted_at=at, keys=keys,
                    k0_left=k0_left, root_index=root_index,
                    roots_burned=list(roots_burned),
                    roots_remaining=n_k0_roots - 1 - root_index,
                    burned_attempts=burned_attempts,
                    near_misses=list(near_misses),
                    min_surplus=float(min(
                        (r.extras["surplus"] for r in rounds), default=np.nan)),
                    reason=reason)

    n = 1
    while n <= n_rounds:
        fire = (tamper_at == n and tamper_budget > 0)
        if fire:
            tamper_budget -= 1
        rr = simulate_round(n, seed_bytes, n_symbols, cfg_cv, cfg_cdma, cfg_bud,
                            distance_km, interferer_powers, eps_mui,
                            tamper=fire, auth_key=auth_key,
                            prev_codebook=prev_codebook,
                            rate_model=rate_model, eps_sm=eps_sm, eps_pa=eps_pa)
        # --- PE rejection (modelled): estimated excess noise beyond the
        # eps_pe confidence bound -> the design noise was wrong, abort.
        if pe_reject_at == n:
            rr = RoundResult(round_index=rr.round_index, k_bits=rr.k_bits,
                             a_bits=rr.a_bits, c_bits=rr.c_bits,
                             u_bits=rr.u_bits, auth_ok=rr.auth_ok,
                             xi_total=rr.xi_total, xi_mui=rr.xi_mui,
                             rate_per_symbol=rr.rate_per_symbol,
                             feasible=rr.feasible, codebook=rr.codebook,
                             extras=dict(rr.extras, pe_rejected=True))
            rounds.append(rr)
            return _finish("parameter estimation rejected", n)
        if n == 1:
            # bootstrap: auth key AND code seed come out of the root
            k0_left -= rr.a_bits + rr.c_bits

        # --- budget infeasibility (u_n < 0): cannot pay auth + seed ---
        if enforce_budget and not rr.feasible:
            rounds.append(rr)
            return _finish("budget infeasible (u_n < 0)", n)

        if not rr.auth_ok:
            can_retry = (recover and n == 1
                         and root_index + 1 < n_k0_roots)
            if not can_retry:
                rounds.append(rr)      # the aborting round belongs to the run
                return _finish("authentication failure (tamper detected)", n)
            # burn the exposed root and restart round 1 under the next one
            burned_attempts.append(rr)
            roots_burned.append(root_index)
            root_index += 1
            k0_key = _root_key(root_index)
            auth_key = kdf_expand(k0_key, TAG_AUTH, 32)
            seed_bytes = kdf_expand(k0_key, TAG_SEED, 32)
            prev_codebook = None
            continue      # retry the same round index with the fresh root

        # A near miss is *feasible but tight*: it survives yet has almost no
        # surplus.  Infeasible rounds are a different (already-fatal) case, so
        # they must not be lumped in here.
        if rr.feasible and rr.extras["surplus"] < near_miss_ratio:
            near_misses.append(rr.round_index)

        rounds.append(rr)
        keys.append(rr.u_bits)
        # ---- rolling: K_n -> a_{n+1}, c_{n+1}, u_n --------------------- #
        # Round n+1's auth key and code seed come from *this* round's key
        # K_n -- NOT from c_n.  Deriving them from the seed would make
        # authentication (and the next seed) depend on code secrecy, i.e.
        # a_i = f(c_{i-1}); since c_{i-1} is exactly what an attacker tries
        # to blind-estimate (docs/03), that would put the whole auth chain
        # downstream of code leakage.  Splitting K_n keeps the two failure
        # modes independent, which is the point of the three-way split.
        if rr.feasible:
            nxt = split_key(rr.k_bits, cfg_bud,
                            overhead_bits=rr.a_bits,
                            key_material=rr.extras["key_material"])
            auth_key = nxt["a"]
            seed_bytes = nxt["c"]
        elif enforce_budget:
            raise AssertionError("unreachable: infeasible rounds abort above")
        else:
            # Studying the infeasible region on purpose: there is not enough
            # key to derive the next round, so the chain cannot continue.
            # Stop here rather than inventing material.
            return _finish("budget infeasible (u_n < 0)", n)
        prev_codebook = rr.codebook
        n += 1

    return _finish("completed", None)


def loop_summary(result: dict) -> dict:
    """Aggregate statistics of a loop run."""
    rounds = result["rounds"]
    if not rounds:
        return {}
    k = np.array([r.k_bits for r in rounds])
    u = np.array([r.u_bits for r in rounds])
    a = np.array([r.a_bits for r in rounds])
    return dict(n_rounds=len(rounds), aborted_at=result["aborted_at"],
                K_total=float(k.sum()), u_total=float(np.clip(u, 0, None).sum()),
                a_total=float(a.sum()),
                overhead=float((a.sum() + rounds[0].c_bits * len(rounds)) / k.sum())
                if k.sum() > 0 else np.inf,
                min_surplus=float(min(r.extras["surplus"] for r in rounds)),
                xi_mui=rounds[0].xi_mui,
                rate_per_symbol=rounds[0].rate_per_symbol,
                code_model=rounds[0].extras.get("code_model", "unknown"),
                refresh_fraction=rounds[0].extras.get("refresh_fraction", 1.0),
                auth_ok_all=all(r.auth_ok for r in rounds))
