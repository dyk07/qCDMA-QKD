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
                                model: str = "extractor") -> dict:
    r"""Minimum refresh fraction f, from the seed's min-entropy budget.

    Derivation
    ----------
    Chips are re-drawn independently with probability ``f``, so a fraction
    ``1 - f`` of the round-``n`` code equals round-``n-1``'s code.  If Eve
    succeeded on a fraction ``p_blind`` of the *previous* code, she knows

        k_known = (1 - f) * L * p_blind

    chips of the current code.  Under the extractor model each known chip is one
    linear constraint on the ``seed_bits``-bit seed, so a non-zero min-entropy
    of the current seed requires the *known* chips to be fewer than the seed:

        (1 - f) * L * p_blind < seed_bits
        <=>  f > 1 - seed_bits / (L * p_blind)

    For the defaults (``seed_bits = 256``, ``L = 1024``, ``p_blind = 1``) this
    gives ``f > 0.75`` -- far stricter than the naive "fresh chips must outnumber
    the seed" rule (``f > 0.25``), which is **wrong** because what matters is how
    much of the seed Eve *knows*, not how much she does not.

    This is the concrete, previously unstated design constraint of the paper:
    the spreading factor, the code-seed length and the refresh interval are not
    independent knobs.  Alternatively one shortens the seed (``L / seed_bits``
    chips per seed bit) or accepts the computational-PRF model, which has no such
    constraint at the cost of a computational security statement.
    """
    if L <= 0:
        raise ValueError("L must be positive")
    if model == "prf":
        return dict(model=model, seed_bits=int(seed_bits), L=int(L),
                    p_blind=float(p_blind), f_min=0.0,
                    max_known_chips=float(L), feasible=True,
                    note="computational PRF: no extractor constraint")
    if model != "extractor":
        raise ValueError(f"unknown model {model!r}")
    # infeasible as soon as the seed itself exceeds the code length
    if seed_bits >= L * float(p_blind):
        return dict(model=model, seed_bits=int(seed_bits), L=int(L),
                    p_blind=float(p_blind), f_min=1.0,
                    max_known_chips=float(seed_bits),
                    known_chips_at_fmin=float(seed_bits),
                    feasible=False,
                    note="seed longer than the code: no refresh rate can help")
    f_min = float(np.clip(1.0 - seed_bits / (L * float(p_blind)), 0.0, 1.0))
    return dict(model=model, seed_bits=int(seed_bits), L=int(L),
                p_blind=float(p_blind), f_min=f_min,
                max_known_chips=float(seed_bits),
                known_chips_at_fmin=float(seed_bits),
                feasible=bool(f_min < 1.0),
                note="extractor: known chips must not exhaust the seed")


def max_chips_per_seed_bit(seed_bits: int, refresh_fraction: float,
                           p_blind: float = 1.0) -> float:
    """Largest ``L / seed_bits`` allowed at a given refresh fraction.

    Inverting ``(1-f) L p < seed_bits`` gives ``L < seed_bits / ((1-f) p)``,
    i.e. at most ``1/((1-f) p)`` chips per seed bit.
    """
    denom = max(1.0 - float(refresh_fraction), 1e-12) * float(p_blind)
    return float(1.0 / denom)


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
