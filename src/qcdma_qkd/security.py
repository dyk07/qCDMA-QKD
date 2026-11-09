"""Composable security-parameter bookkeeping.

Components
----------
eps_qkd   : CV-QKD protocol (finite-size, smooth min-entropy + PA)
eps_sm    : smoothness of the min-entropy estimate
eps_pe    : parameter estimation confidence
eps_pa    : privacy amplification
eps_auth  : classical authentication (per round, summed over messages)
eps_code  : code-distribution reduction (MitM replaced / replacement detected)
eps_mui   : dynamic-MUI outlier event
eps_prf   : PRF / extractor distinguishing advantage (if computational)

Composition
-----------
Basic (union bound, trace distance):
    eps_total <= sum_i eps_i

Sequential composition over N rounds (diamond norm, "advanced"):
    eps_total <= sqrt(2 N ln(1/eps_delta)) * eps_round + eps_delta

Leftover-hash-lemma budget for the KDF (information-theoretic extractor):
    m_out <= H_min - 2 log2(1/eps_kdf)       [bits]

Guessing / blind-estimation bound for the code seed (Section 5):
    eps_code <= 2^{-(H_min(c_n) - m_code)} * poly(...)
    modelled as  P_guess = 2^{-(1-f) * seed_bits}
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .config import SecurityConfig


# --------------------------------------------------------------------------- #
# elementary composition
# --------------------------------------------------------------------------- #
def compose_sum(eps: dict | list | np.ndarray) -> float:
    """Union-bound composition."""
    vals = list(eps.values()) if isinstance(eps, dict) else list(eps)
    return float(np.sum(vals))


def compose_sequential(eps_round: float, n_rounds: int,
                       eps_delta: float = 1e-12) -> float:
    """Diamond-norm sequential composition bound."""
    if n_rounds <= 0:
        return 0.0
    return float(math.sqrt(2.0 * n_rounds * math.log(1.0 / eps_delta)) * eps_round
                 + eps_delta)


def compose_log(eps_list) -> float:
    """Composition in the ``-log2`` domain (useful for plotting budgets)."""
    return float(-math.log2(max(compose_sum(eps_list), 1e-300)))


# --------------------------------------------------------------------------- #
# extractor / PRF budgets
# --------------------------------------------------------------------------- #
def lhl_output_bits(h_min_bits: float, eps_kdf: float) -> float:
    """Max extractable output length from the leftover hash lemma."""
    return float(max(h_min_bits - 2.0 * math.log2(1.0 / eps_kdf), 0.0))


def lhl_epsilon(h_min_bits: float, out_bits: float) -> float:
    """Achievable epsilon for a given (H_min, output length) pair."""
    return float(2.0 ** (-(h_min_bits - out_bits) / 2.0)) if out_bits < h_min_bits else 1.0


def prf_distinguishing_advantage(seed_bits: int, query_budget: float) -> float:
    """Toy PRF bound: adv <= 2^{-seed_bits/2} + q/2^{seed_bits}."""
    return float(2.0 ** (-seed_bits / 2.0) + query_budget * 2.0 ** (-seed_bits))


def code_leak_epsilon(seed_bits: int, leaked_fraction: float) -> float:
    """P(Eve reconstructs a code seed from a leaked fraction of it)."""
    f = float(np.clip(leaked_fraction, 0.0, 1.0))
    return float(2.0 ** (-(1.0 - f) * seed_bits))


# --------------------------------------------------------------------------- #
# combined budget
# --------------------------------------------------------------------------- #
@dataclass
class EpsBudget:
    eps_qkd: float
    eps_sm: float
    eps_pe: float
    eps_pa: float
    eps_auth: float
    eps_code: float
    eps_mui: float
    eps_prf: float

    def total(self) -> float:
        return compose_sum(asdict(self))

    def table(self) -> dict:
        d = asdict(self)
        d["total"] = self.total()
        return d


def from_config(cfg: SecurityConfig, eps_qkd_multiplier: float = 1.0,
                eps_auth_rounds: int = 1, eps_prf: float = 1e-12) -> EpsBudget:
    """Build an :class:`EpsBudget` describing **one round**.

    ``eps_auth_rounds`` multiplies ``eps_auth`` and defaults to **1**, not 4.

    Why 1 is correct here.  The eps-AXU bound already charges for repeated use
    of one key::

        eps_AXU <= r * 2^(-t) + 2^(-(l-t)/2),      l = t + 2 log2(1/eps)

    With ``t = 128``, ``eps = 1e-12`` and ``r = 2^20`` the first term is
    ``2^20 * 2^-128 = 3.1e-33``, i.e. far below the second, which equals
    ``eps`` by construction.  A round sends ``n_msg = 667 <= r`` messages under
    **one** freshly derived key, so the per-round forgery probability is simply
    ``eps``.  Multiplying by the message count counts the same reuse twice.

    What the returned total means
    -----------------------------
    ``EpsBudget.total()`` is a **per-round** figure.  Composing it over many
    rounds is a separate step and must not be done by feeding a single
    component back in -- see :func:`compose_over_rounds`.
    """
    return EpsBudget(
        eps_qkd=cfg.eps_qkd * eps_qkd_multiplier,
        eps_sm=cfg.eps_sm,
        eps_pe=cfg.eps_qkd,
        eps_pa=cfg.eps_pa,
        eps_auth=cfg.eps_auth * eps_auth_rounds,
        eps_code=cfg.eps_code,
        eps_mui=cfg.eps_mui,
        eps_prf=eps_prf,
    )


def compose_over_rounds(eps_round: float, n_rounds: int,
                        method: str = "best",
                        eps_delta: float = 1e-12) -> float:
    """Compose a **per-round total** epsilon over ``n_rounds`` rounds.

    ``eps_round`` must be the *whole* per-round budget (:meth:`EpsBudget.total`),
    not one component.  An earlier version of the docs composed ``1e-12`` -- the
    authentication term alone -- and thereby dropped ``eps_qkd``, ``eps_sm`` and
    ``eps_pe`` (``1e-10`` each), understating the composed figure by ~300x at
    ``N = 1e4``.

    ``method``
        ``"union"``      -- union bound, ``N * eps_round``.
        ``"sequential"`` -- diamond-norm composition,
        ``sqrt(2 N ln(1/eps_delta)) * eps_round + eps_delta``.
        ``"best"`` (default) -- ``min`` of the two.

    Why ``min`` is the sound choice
    -------------------------------
    Both forms are *proven* upper bounds on the same quantity, so the truth is
    below both and therefore below their minimum.  Claiming the minimum is both
    sound and the tightest statement available.  (An earlier version defaulted
    to the sequential form alone; at ``N = 1`` its additive ``eps_delta`` makes
    it *looser* than the trivial union bound, so the composed figure looked
    worse than the per-round budget for no reason.)

    The two cross near ``N ~ 50``: below that the union bound wins, above it the
    sequential one does.
    """
    if n_rounds <= 0:
        return 0.0
    seq = compose_sequential(eps_round, n_rounds, eps_delta)
    uni = float(n_rounds * eps_round)
    if method == "sequential":
        return seq
    if method == "union":
        return uni
    if method == "best":
        return min(seq, uni)
    raise ValueError(
        "method must be 'best', 'sequential' or 'union', got %r" % (method,))


def required_round_eps(n_rounds: int, eps_target: float,
                       method: str = "best",
                       eps_delta: float = 1e-12) -> float:
    """Per-round total epsilon needed to keep ``n_rounds`` under the target.

    Inverts :func:`compose_over_rounds`.  ``method="union"`` gives
    ``eps_target / N``; ``method="sequential"`` gives
    ``(eps_target - eps_delta) / sqrt(2 N ln(1/eps_delta))``; the default
    ``"best"`` returns the larger (i.e. the easier requirement, since either
    bound suffices to meet the target).
    """
    if n_rounds <= 0:
        return float(eps_target)
    seq = float(max(eps_target - eps_delta, 0.0)
                / math.sqrt(2.0 * n_rounds * math.log(1.0 / eps_delta)))
    uni = float(eps_target / n_rounds)
    if method == "sequential":
        return seq
    if method == "union":
        return uni
    if method == "best":
        return max(seq, uni)
    raise ValueError(
        "method must be 'best', 'sequential' or 'union', got %r" % (method,))


def max_rounds(eps_round: float, eps_target: float,
               eps_delta: float = 1e-12,
               method: str = "best") -> float:
    """Largest round count whose composed epsilon stays under the target.

    Exact inverse of :func:`compose_over_rounds` (same ``method``), obtained by
    bisection, so the two can never disagree.  An earlier version took a
    ``max`` of two inversions while ``compose_over_rounds`` returned a ``min``,
    which produced the self-contradictory pair "at most 3.28 rounds" and
    "3 rounds already exceeds the target".

    Sanity anchors (per-round ``3.05e-10``, target ``1e-9``): 3 rounds pass,
    4 do not.
    """
    if eps_round <= 0:
        return float(np.inf)
    if compose_over_rounds(eps_round, 1, method, eps_delta) > eps_target:
        return 0.0
    lo, hi = 1.0, 2.0
    while (compose_over_rounds(eps_round, int(hi), method, eps_delta)
           <= eps_target and hi < 1e12):
        lo, hi = hi, hi * 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if compose_over_rounds(eps_round, int(mid), method, eps_delta) <= eps_target:
            lo = mid
        else:
            hi = mid
    return float(np.floor(lo))


def required_log_inv_eps(k_bits: float, h_min_bits: float,
                         max_output_fraction: float = 0.9) -> float:
    """Largest ``log2(1/eps)`` such that the LHL still allows the output.

    Solves  m = H_min - 2 log2(1/eps)  with  m = max_output_fraction * K.
    """
    m = max_output_fraction * k_bits
    slack = h_min_bits - m
    if slack <= 0:
        return 0.0
    return float(slack / 2.0)


def print_budget(b: EpsBudget) -> str:
    """Human-readable table (also used by the docs generator)."""
    d = b.table()
    lines = ["| component | epsilon | -log2 |", "|---|---|---|"]
    for k, v in d.items():
        if k == "total":
            continue
        lines.append(f"| {k} | {v:.1e} | {-math.log2(max(v, 1e-300)):.1f} |")
    lines.append(f"| **total** | **{d['total']:.1e}** | "
                 f"**{-math.log2(max(d['total'], 1e-300)):.1f}** |")
    return "\n".join(lines)
