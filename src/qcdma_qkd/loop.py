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


def split_key(k_bits: float, cfg: BudgetConfig, overhead_bits: float = 0.0) -> dict:
    """Three-way KDF split with independent domain tags.

    ``overhead_bits`` is the authentication cost that must be paid out of the
    *next* round's key (typically ``budget.round_auth_bits``).  Returns byte
    strings for the auth key, the code seed and the application output, each
    derived from the same root with a distinct domain tag, so they are
    independent under the PRF/KDF assumption.
    """
    a_bits = float(overhead_bits)
    c_bits = float(cfg.seed_bits)
    u_bits = float(k_bits) - a_bits - c_bits
    if u_bits < 0:
        raise ValueError(f"key exhausted: K={k_bits} < a+c={a_bits + c_bits}")
    root = hashlib.sha256(f"K-{k_bits}".encode()).digest()

    def _bytes(bits: float) -> int:
        return int(np.ceil(max(bits, 1) / 8.0))

    return dict(
        a_bits=a_bits, c_bits=c_bits, u_bits=u_bits,
        a=kdf_expand(root, TAG_AUTH, _bytes(a_bits)),
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
    """
    rows = np.empty((n_users, L), dtype=np.float64)
    for i in range(n_users):
        stream = kdf_expand(seed, TAG_SEED + f"/u{i}/f{frame}/s{session}".encode(),
                            int(np.ceil(L / 8.0)))
        bits = np.unpackbits(np.frombuffer(stream, dtype=np.uint8))[:L]
        rows[i] = 2.0 * bits - 1.0
    if base is not None and float(refresh_fraction) < 1.0:
        rng = rng if rng is not None else np.random.default_rng(frame)
        rows = codes.refreshed_codebook(base, refresh_fraction, rng)
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
                   prev_codebook: np.ndarray | None = None) -> RoundResult:
    """One closed-loop round with the real budget and MUI accounting.

    ``K_n`` is *computed*, not supplied: it is the raw post-PA key produced by
    the quantum block, ``rate_per_symbol * n_symbols``.  The auth key is passed
    in explicitly because it comes from the *previous* round.
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
    r = cvqkd.asymptotic_rate(cfg_cv.V_A, T, xi_tot, cfg_cv.eta, cfg_cv.v_el,
                              cfg_cv.beta, cfg_cv.detection)
    i_ab = r["I_AB"]
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
    return RoundResult(round_index=round_index, k_bits=float(k_bits),
                       a_bits=b["a"], c_bits=b["c"], u_bits=b["u"],
                       auth_ok=auth_ok, xi_total=xi_tot, xi_mui=xi_mui,
                       rate_per_symbol=r["S_pos"], feasible=b["feasible"],
                       codebook=codebook,
                       extras=dict(I_AB=i_ab, chi_BE=r["chi_BE"],
                                   n_messages=b["n_messages"],
                                   surplus=b["surplus"], T=T,
                                   seed=seed, auth_key=auth_key,
                                   code_model=cfg_cdma.code_model,
                                   refresh_fraction=cfg_cdma.refresh_fraction))


# --------------------------------------------------------------------------- #
# multi-round loop
# --------------------------------------------------------------------------- #
def run_loop(n_rounds: int, k0_bits: int, cfg_cv: CVQKDConfig,
             cfg_cdma: CDMAConfig, cfg_bud: BudgetConfig,
             distance_km: float, n_symbols: int = 10 ** 8,
             eps_mui: float = 1e-12,
             interferer_powers: np.ndarray | None = None,
             tamper_at: int | None = None) -> dict:
    """Run the closed loop, bootstrapping from K_0.

    Accounting model (the point of the whole exercise):

    * round 1 authenticates with ``K_0``; thereafter with ``a_{n-1}``, a
      sub-key of round n-1's key.  The *root* is never reused.
    * ``K_n`` is regenerated from scratch by the quantum block, so the loop does
      **not** consume net output in steady state -- it consumes only the
      ``a_n + c_n`` slice of each round's own key.
    """
    rounds: list[RoundResult] = []
    k0_left = float(k0_bits)
    keys = []
    k0_key = hashlib.sha256(b"K0-root").digest()
    auth_key = kdf_expand(k0_key, TAG_AUTH, 32)
    seed_bytes = kdf_expand(k0_key, TAG_SEED, 32)
    prev_codebook: np.ndarray | None = None

    for n in range(1, n_rounds + 1):
        rr = simulate_round(n, seed_bytes, n_symbols, cfg_cv, cfg_cdma, cfg_bud,
                            distance_km, interferer_powers, eps_mui,
                            tamper=(tamper_at == n), auth_key=auth_key,
                            prev_codebook=prev_codebook)
        rounds.append(rr)
        if n == 1:
            # bootstrap: auth key AND code seed come out of K_0
            k0_left -= rr.a_bits + rr.c_bits
        if not rr.auth_ok:
            return dict(rounds=rounds, aborted_at=n, keys=keys,
                        k0_left=k0_left,
                        reason="authentication failure (tamper detected)")
        keys.append(rr.u_bits)
        # rolling: round n+1 authenticates under a sub-key of round n's key
        auth_key = kdf_expand(rr.extras["seed"], TAG_AUTH, 32)
        seed_bytes = kdf_expand(rr.extras["seed"], TAG_SEED, 32)
        prev_codebook = rr.codebook

    return dict(rounds=rounds, aborted_at=None, keys=keys, k0_left=k0_left,
                reason="completed")


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
