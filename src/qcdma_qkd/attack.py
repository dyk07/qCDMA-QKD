"""Adversarial models for the q-CDMA code-distribution loop.

Three attacks are modelled, all at the *classical* layer where the paper's
contribution lies:

A. ``man_in_the_middle``
   Eve intercepts the code-update message and runs two independent, perfectly
   legitimate CV-QKD sessions -- one with Alice, one with Bob.  Both sessions
   pass parameter estimation, because PE only checks the *channel statistics*,
   not the peer identity.  Alice's final key ``K_AE`` and Bob's ``K_BE`` are
   independent, so the end-to-end keys differ.  This is the attack that an
   unauthenticated code-distribution channel enables.

   ``undetected_probability_unauthenticated`` returns 1.0 - PE never notices.
   With authentication the tamper is caught with probability
   ``1 - forgery_probability``.

B. ``blind_code_estimation``
   Eve observes the multiplexed optical field and tries to recover the
   spreading code of user i.  Modelled by a per-chip success probability
   ``p_chip`` derived from an effective observation SNR; the resulting
   min-entropy of the code seed is propagated to ``eps_code``.

C. ``cross_round_leakage``
   Eve's partial knowledge of round ``n-1``'s code is combined with the PRF
   structure of round n.  This yields the *key design constraint* of the paper:

       IT-extractor codes:   f * L > seed_bits      (f = refresh fraction)
       computational PRF:    no constraint (security degrades to computational)

   i.e. a too-large refresh interval, or a too-short seed, silently destroys
   the access-control property even though the code "looks" secret.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# A. Man-in-the-middle on the code-distribution channel
# --------------------------------------------------------------------------- #
@dataclass
class MitMResult:
    end_to_end_keys_match: bool
    pe_passes_alice_eve: bool
    pe_passes_bob_eve: bool
    detected_by_auth: bool
    detection_probability: float
    rounds_until_detection: float
    note: str


def man_in_the_middle(tag_bits: int = 128, authenticated: bool = True,
                      n_messages_per_round: int = 8,
                      rounds_per_second: float = 1.0) -> MitMResult:
    """Two-session MitM against the code-update message.

    ``authenticated=False``: Eve replaces the code seed with her own; both
    Alice and Bob run valid QKD with Eve and both accept.  Since the two
    sessions are independent, the end-to-end keys differ and the *first*
    application message fails -- but only after the round has been completed.

    ``authenticated=True``: Eve must forge the MAC on the seed under ``a_n``.
    With an eps-AXU family ``P_forge <= 2^{-tag_bits}``; across all messages of
    a round the union bound gives ``1 - n_msgs * 2^{-t}`` detection.
    """
    p_forge = float(min(1.0, n_messages_per_round * 2.0 ** (-tag_bits)))
    if not authenticated:
        return MitMResult(
            end_to_end_keys_match=False,
            pe_passes_alice_eve=True,
            pe_passes_bob_eve=True,
            detected_by_auth=False,
            detection_probability=0.0,
            rounds_until_detection=np.inf,
            note=("undetected indefinitely at the PE layer; the loop silently "
                  "produces two disjoint keys"),
        )
    p_det = 1.0 - p_forge
    return MitMResult(
        end_to_end_keys_match=False,
        pe_passes_alice_eve=True,
        pe_passes_bob_eve=True,
        detected_by_auth=True,
        detection_probability=p_det,
        rounds_until_detection=1.0 / p_det if p_det > 0 else np.inf,
        note=("seed tamper rejected with prob 1 - n*2^-t; the loop aborts "
              "before any quantum resources are spent"),
    )


def detection_after_k_rounds(tag_bits: int, n_messages_per_round: int,
                             rounds: int) -> float:
    """P(MitM still undetected after ``rounds`` rounds).

    Each round is an independent forgery attempt, so the survival probability
    decays geometrically: ``(n 2^{-t})^rounds``.
    """
    p_forge = min(1.0, n_messages_per_round * 2.0 ** (-tag_bits))
    return float(p_forge ** max(int(rounds), 0))


# --------------------------------------------------------------------------- #
# B. Blind code estimation
# --------------------------------------------------------------------------- #
def chip_success_probability(observation_snr: float, L: int,
                             n_observations: float) -> float:
    """Per-chip success probability of blind code estimation.

    A single observation of one symbol gives an SNR of ``observation_snr / L``
    per chip (the interferer's symbol is unknown and acts as noise); averaging
    ``n_observations`` independent symbols gives

        SNR_chip = observation_snr * n_observations / L
        p_chip   = Phi(sqrt(SNR_chip))

    (BPSK chip, matched filter, Gaussian approximation).  This is a *model*, not
    an information-theoretic bound -- the true blind-despreading bound is the
    open problem flagged in the paper outline.
    """
    from scipy import stats
    snr = max(observation_snr, 0.0) * float(n_observations) / max(L, 1)
    if snr <= 0:
        return 0.5
    return float(stats.norm.cdf(np.sqrt(snr)))


def residual_seed_entropy(seed_bits: int, L: int, p_chip: float,
                          model: str = "extractor") -> float:
    """Min-entropy of the code seed remaining to Eve.

    ``extractor`` (information-theoretic, e.g. Toeplitz / one-time-pad style):
        every correctly estimated chip is one linear constraint on the seed.
        With ``k`` correct chips and a random ``k x seed_bits`` matrix whose
        rank saturates at ``min(k, seed_bits)``:

            H_min = max(0, seed_bits - k),      k ~ Binomial(L, p_chip)

        so the seed is fully exposed once Eve gets ``seed_bits`` chips right.
        We use the mean ``k = p_chip * L`` (concentration is tight at L ~ 1e3).

    ``prf`` (computational PRF): output bits do not leak seed bits, so

            H_min = seed_bits  (as long as the seed itself was never leaked)

        at the price of a computational assumption in the security statement.
    """
    if model == "prf":
        return float(seed_bits)
    if model != "extractor":
        raise ValueError(f"unknown model {model!r}")
    k = float(p_chip) * float(L)
    return float(max(0.0, seed_bits - k))


def eps_code_from_entropy(h_min: float, security_margin_bits: float = 0.0) -> float:
    """Guessing probability of the seed: 2^{-H_min} (capped at 1)."""
    return float(min(1.0, 2.0 ** (-(h_min - security_margin_bits))))


def required_entropy_bits(eps_code: float) -> float:
    """Min-entropy (bits) that must survive for a given eps_code target.

        eps_code = 2^{-H_min}  <=>  H_min = log2(1/eps_code)

    This is the ``k_req`` of the refresh constraint: it is the amount of seed
    entropy that must remain *unconsumed*.  "H_min > 0" is NOT a security
    criterion -- it only excludes the single point where Eve is certain, and
    still permits eps_code = 0.5 (one bit of entropy).  Every security statement
    in this project is written in terms of an explicit eps_code target.
    """
    if not 0.0 < eps_code <= 1.0:
        raise ValueError("eps_code must be in (0, 1]")
    return float(np.log2(1.0 / eps_code))


def max_tolerable_known_chips(seed_bits: int, eps_code: float = 1e-12) -> float:
    """Largest number of chips Eve may know while keeping eps_code on target.

        H_min = seed_bits - k_known >= k_req = log2(1/eps_code)
        =>  k_known <= seed_bits - k_req

    With seed=256, eps=1e-12 this is 216.1 chips, not 256: the last ~40 bits of
    the seed are the security margin and cannot be spent.
    """
    return float(seed_bits - required_entropy_bits(eps_code))


def required_observations(seed_bits: int, L: int, observation_snr: float,
                          target_p_chip: float = 0.99) -> float:
    """Observations Eve needs to reach a per-chip success of ``target_p_chip``."""
    from scipy import stats
    z = stats.norm.isf(1.0 - target_p_chip)      # upper quantile
    snr_needed = z ** 2
    if observation_snr <= 0:
        return np.inf
    return float(snr_needed * L / observation_snr)


def blind_estimation_table(observation_snrs: np.ndarray, seed_bits: int, L: int,
                           n_observations_list=(1e3, 1e6, 1e9),
                           model: str = "extractor") -> dict:
    """Grid of (observation SNR, #observations) -> H_min and eps_code."""
    out = dict(snr=[], n_obs=[], p_chip=[], h_min=[], eps_code=[])
    for snr in np.atleast_1d(observation_snrs):
        for n_obs in n_observations_list:
            p = chip_success_probability(float(snr), L, float(n_obs))
            h = residual_seed_entropy(seed_bits, L, p, model)
            out["snr"].append(float(snr))
            out["n_obs"].append(float(n_obs))
            out["p_chip"].append(p)
            out["h_min"].append(h)
            out["eps_code"].append(eps_code_from_entropy(h))
    return {k: np.asarray(v) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# C. Cross-round leakage and the refresh constraint
# --------------------------------------------------------------------------- #
def refresh_security_constraint(seed_bits: int, L: int,
                                p_blind: float = 1.0,
                                eps_code: float = 1e-12,
                                model: str = "extractor") -> dict:
    r"""Minimum refresh fraction f, from an explicit eps_code target.

    Security criterion
    ------------------
    The *wrong* criterion (used by earlier versions) is ``H_min > 0``.  That only
    excludes the single point where Eve is certain, and still permits
    ``eps_code = 0.5``.  The correct requirement is

        eps_code = 2^{-H_min} <= eps_code_target
        <=>  H_min >= k_req := log2(1/eps_code_target)

    With ``H_min = seed_bits - k_known`` this gives

        k_known <= seed_bits - k_req

    so the last ``k_req`` bits of the seed are a security margin that cannot be
    spent.  With ``seed = 256`` and ``eps = 1e-12``, ``k_req = 39.9`` bit, so the
    tolerable leakage is 216.1 chips, not 256.

    Mean (closed form)
    ------------------
    Chips are re-drawn independently with probability ``f``, so a fraction
    ``1 - f`` of the round-n code equals round-(n-1)'s code.  If Eve knows a
    fraction ``p_blind`` of the previous code, she knows

        k_known = (1 - f) * L * p_blind

    chips of the current code.  Requiring ``k_known <= seed_bits - k_req``:

        (1 - f) * L * p_blind < seed_bits - k_req
        <=>  f > 1 - (seed_bits - k_req) / (L * p_blind)

    This is the **mean** rule: it constrains E[k_known].  It is only exact when
    ``p_blind = 1``; for ``p_blind < 1`` use
    :func:`refresh_constraint_tail`, which is the default everywhere else in
    this project.

    Boundary behaviour
    ------------------
    Since ``f = 1`` gives ``k_known = 0``, a feasible ``f`` exists whenever
    ``seed_bits > k_req``.  The constraint *stops binding* (``f_min = 0``) once
    ``L * p_blind <= seed_bits - k_req``: a seed long enough relative to the code
    needs no refreshing at all.  There is no "infeasible" regime -- only a
    regime where the constraint is vacuous -- **unless** ``seed_bits <= k_req``,
    i.e. the seed is too short to even hold the security margin.  That case is
    genuinely infeasible and is reported as such.
    """
    if L <= 0:
        raise ValueError("L must be positive")
    if not 0.0 < eps_code <= 1.0:
        raise ValueError("eps_code must be in (0, 1]")
    k_req = required_entropy_bits(eps_code)
    p = float(np.clip(p_blind, 0.0, 1.0))
    base = dict(model=model, seed_bits=int(seed_bits), L=int(L),
                p_blind=p, eps_code=eps_code, k_req=k_req)

    if model == "prf":
        return {**base, "f_min": 0.0, "max_known_chips": float(L),
                "feasible": True, "binds": False,
                "note": "computational PRF: no extractor constraint"}

    if model != "extractor":
        raise ValueError(f"unknown model {model!r}")

    # the seed must be able to hold the security margin at all
    if seed_bits <= k_req:
        return {**base, "f_min": 1.0,
                "max_known_chips": float(seed_bits - k_req),
                "feasible": False, "binds": True,
                "note": ("seed_bits <= log2(1/eps_code): the seed is too short "
                         "to hold the security margin; lengthen the seed")}

    budget_chips = seed_bits - k_req        # entropy that may be consumed
    if p <= 0.0:
        return {**base, "f_min": 0.0, "max_known_chips": float(budget_chips),
                "feasible": True, "binds": False,
                "note": "p_blind = 0: no leakage, no constraint"}

    f_min = float(np.clip(1.0 - budget_chips / (L * p), 0.0, 1.0))
    return {**base, "f_min": f_min, "max_known_chips": float(budget_chips),
            "known_chips_at_fmin": float(budget_chips),
            "feasible": True, "binds": bool(f_min > 0.0),
            "note": ("extractor: known chips must not consume the security "
                     "margin" if f_min > 0 else
                     "constraint does not bind (f = 0 already meets eps_code)")}


def refresh_constraint_tail(seed_bits: int, L: int, p_blind: float = 1.0,
                            eps_code: float = 1e-12,
                            tol: float = 1e-4) -> dict:
    r"""Default refresh constraint: exact binomial **tail**, eps_code target.

    This is the rigorous form and should be preferred over the closed form.

    Two corrections over :func:`refresh_security_constraint`
    -------------------------------------------------------
    (a) **eps target, not H_min > 0.**  The tail event of interest is that Eve
        consumes the whole security margin, not merely the whole seed:

            k_known >= seed_bits - k_req,      k_req = log2(1/eps_code)

    (b) **Tail, not mean.**  ``k_known ~ Binomial(round((1-f) L), p_blind)`` is
        random; constraining its mean leaves the upper tail uncontrolled.

    The solver is therefore

        f_min = min{ f : P(k_known >= seed_bits - k_req) <= eps_code_tail }

    where ``eps_code_tail`` is split from the budget: we require the tail
    probability itself to be at most ``eps_code``, which is then a *total*
    variation distance of the same order.

    Numbers (seed=256, L=1024, p=1, eps=1e-12): ``k_req = 39.9``, so at most
    216.1 chips may be known, giving ``f_min = 1 - 216.1/1024 = 0.789`` --
    versus 0.75 under the old ``H_min > 0`` rule, whose actual eps_code at
    f = 0.75 is 1.0 (total loss).
    """
    from scipy import stats

    if L <= 0:
        raise ValueError("L must be positive")
    if not 0.0 < eps_code <= 1.0:
        raise ValueError("eps_code must be in (0, 1]")
    k_req = required_entropy_bits(eps_code)
    k_max = seed_bits - k_req          # chips may be known without violating eps
    p = float(np.clip(p_blind, 0.0, 1.0))
    base = dict(seed_bits=int(seed_bits), L=int(L), p_blind=p,
                eps_code=eps_code, k_req=k_req, k_max_allowed=float(k_max))

    if seed_bits <= k_req:
        return {**base, "f_min": 1.0, "feasible": False, "violation": 1.0,
                "note": "seed too short to hold the security margin"}

    if p <= 0.0:
        return {**base, "f_min": 0.0, "feasible": True, "violation": 0.0,
                "note": "p_blind = 0: no leakage"}

    def violation(f: float) -> float:
        """P(k_known >= k_max) for a refresh fraction f."""
        n_reused = int(round((1.0 - f) * L))
        # known chips cannot exceed the reused chips
        if n_reused <= k_max:
            return 0.0
        if p >= 1.0:
            return 1.0
        return float(stats.binom.sf(np.ceil(k_max) - 1, n_reused, p))

    if violation(1.0) > eps_code:
        return {**base, "f_min": 1.0, "feasible": False,
                "violation": violation(1.0),
                "note": "no refresh rate satisfies the tail condition"}
    if violation(0.0) <= eps_code:
        return {**base, "f_min": 0.0, "feasible": True,
                "violation": violation(0.0), "binds": False,
                "note": "tail condition already met without refreshing"}

    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if violation(mid) > eps_code:
            lo = mid
        else:
            hi = mid
    return {**base, "f_min": float(hi), "feasible": True,
            "violation": violation(hi), "binds": True,
            "f_min_mean": float(refresh_security_constraint(
                seed_bits, L, p, eps_code)["f_min"]),
            "note": "exact binomial tail constraint at eps_code target"}


def max_chips_per_seed_bit(seed_bits: int, refresh_fraction: float,
                           p_blind: float = 1.0,
                           eps_code: float = 1e-12) -> float:
    """Largest ``L / seed_bits`` allowed at a given refresh fraction.

    Inverting ``(1-f) L p <= seed_bits - k_req`` gives
    ``L <= (seed_bits - k_req) / ((1-f) p)``, i.e. at most
    ``(1 - k_req/seed_bits) / ((1-f) p)`` chips per seed bit.
    """
    budget = seed_bits - required_entropy_bits(eps_code)
    if budget <= 0:
        return 0.0
    denom = max(1.0 - float(refresh_fraction), 1e-12) * float(p_blind)
    return float(budget / seed_bits / denom)


def cross_round_eps(refresh_fraction: float, seed_bits: int, L: int,
                    p_blind: float, model: str = "extractor") -> float:
    """eps_code at round n given round n-1 leakage.

    ``p_blind`` is Eve's per-chip success probability against round n-1.
    """
    p_reused = (1.0 - float(refresh_fraction)) * float(p_blind)
    h = residual_seed_entropy(seed_bits, L, p_reused, model)
    return eps_code_from_entropy(h)


def refresh_sweep(refresh_fractions: np.ndarray, seed_bits: int, L: int,
                  p_blind: float = 0.99, model: str = "extractor") -> dict:
    """eps_code vs refresh fraction, for both extractor and PRF models."""
    out = dict(refresh=[], eps_extractor=[], eps_prf=[], h_extractor=[])
    for f in np.atleast_1d(refresh_fractions):
        f = float(f)
        p_reused = (1.0 - f) * float(p_blind)
        h = residual_seed_entropy(seed_bits, L, p_reused, "extractor")
        out["refresh"].append(f)
        out["h_extractor"].append(h)
        out["eps_extractor"].append(eps_code_from_entropy(h))
        out["eps_prf"].append(2.0 ** (-seed_bits))
    return {k: np.asarray(v) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# D. Quantum-state-preparation coupling (Section 5, second layer)
# --------------------------------------------------------------------------- #
def state_preparation_leak(partial_code_knowledge: float, V_A: float,
                           L: int) -> dict:
    """Extra excess noise Eve can induce by knowing part of the code.

    If the code modulates the *phase* of the quantum state, Eve's partial
    knowledge of ``c_n`` lets her narrow the uncertainty set of the prepared
    ensemble.  Modelled as a correlation between signal and Eve's ancilla of
    strength ``g = partial_code_knowledge``:

        xi_prep = g^2 * V_A / (2 L)

    This term is **not** absorbed by privacy amplification (it is a
    state-preparation effect, not a classical transcript effect), so it must be
    provisioned in the rate -- it is *additive* to the MUI excess noise and must
    be charged to ``eps_code``.  The C4 experiment quantifies how small it must
    be to be negligible.
    """
    g = float(np.clip(partial_code_knowledge, 0.0, 1.0))
    xi = g ** 2 * float(V_A) / (2.0 * float(L))
    return dict(g=g, xi_prep=float(xi))


def max_tolerable_partial_knowledge(xi_budget: float, V_A: float,
                                    L: int) -> float:
    """Largest g with xi_prep <= xi_budget."""
    if V_A <= 0:
        return 0.0
    return float(min(1.0, math.sqrt(max(xi_budget, 0.0) * 2.0 * L / V_A)))
