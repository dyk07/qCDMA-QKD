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
    """Min-entropy of the code seed remaining to Eve, at the **mean** leakage.

    ``extractor`` (information-theoretic, e.g. Toeplitz / one-time-pad style):
        every correctly estimated chip is one linear constraint on the seed.
        With ``k`` correct chips and a random ``k x seed_bits`` matrix whose
        rank saturates at ``min(k, seed_bits)``:

            H_min = max(0, seed_bits - k),      k ~ Binomial(L, p_chip)

        so the seed is fully exposed once Eve gets ``seed_bits`` chips right.

    **This uses the mean ``k = p_chip * L`` and is therefore only a model.**
    ``H_min`` is a *minimum over outcomes*, so it must be quoted at an upper
    tail of ``k``, not at its mean: the mean describes the typical case, and a
    security statement about the typical case is vacuous.  Use
    :func:`residual_seed_entropy_tail` for anything that enters a security
    claim; the gap is quantified by :func:`entropy_gap_vs_tail`.

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


def residual_seed_entropy_tail(seed_bits: int, L: int, p_chip: float,
                               eps_entropy: float = 1e-12,
                               model: str = "extractor") -> float:
    r"""Min-entropy of the code seed at rigorous confidence ``1 - eps``.

    ``k ~ Binomial(L, p_chip)`` is random.  A statement of the form
    "with probability at least ``1 - eps_entropy`` Eve knows at most ``k_star``
    chips" requires ``k_star`` to be an **upper** quantile:

        k_star = min{ k : P(Binomial(L, p_chip) > k) <= eps_entropy }

    and then ``H_min`` is defined by that worst case::

        H_min^{eps} = max(0, seed_bits - k_star)

    Using the mean instead (as :func:`residual_seed_entropy` does) understates
    the leakage, i.e. **overstates** the surviving entropy and hence the
    security.  The difference is small for ``L p`` far from both 0 and
    ``seed_bits`` but it is the difference between a model and a bound, and it
    only ever goes one way.

    ``model="prf"`` behaves as in :func:`residual_seed_entropy`.
    """
    from scipy import stats

    if model == "prf":
        return float(seed_bits)
    if model != "extractor":
        raise ValueError(f"unknown model {model!r}")
    if not 0.0 < eps_entropy < 1.0:
        raise ValueError("eps_entropy must be in (0, 1)")
    n, p = int(L), float(np.clip(p_chip, 0.0, 1.0))
    if n <= 0 or p <= 0.0:
        return float(seed_bits)
    if p >= 1.0:
        return float(max(0.0, seed_bits - n))
    # smallest k with P(X > k) <= eps  <=>  P(X <= k) >= 1 - eps
    k_star = float(stats.binom.ppf(1.0 - eps_entropy, n, p))
    k_star = max(k_star, float(n * p))          # never below the mean
    return float(max(0.0, seed_bits - k_star))


def entropy_gap_vs_tail(seed_bits: int, L: int, p_chip: float,
                        eps_entropy: float = 1e-12) -> dict:
    """Mean-based vs tail-based surviving seed entropy (the size of the model error).

    ``entropy_lost`` is how many *more* bits Eve is credited with at confidence
    ``1 - eps`` than at the mean.  Because ``eps_code = 2^{-H_min}``, each lost
    bit multiplies the guessing probability by 2.
    """
    mean_h = residual_seed_entropy(seed_bits, L, p_chip)
    tail_h = residual_seed_entropy_tail(seed_bits, L, p_chip, eps_entropy)
    return dict(entropy_mean=float(mean_h), entropy_tail=float(tail_h),
                entropy_lost=float(mean_h - tail_h),
                eps_mean=float(eps_code_from_entropy(mean_h)),
                eps_tail=float(eps_code_from_entropy(tail_h)))


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
# B'. What Eve can actually learn: the observation law depends on the codebook
#     only through its Gram matrix -- which splits the problem into TWO regimes
# --------------------------------------------------------------------------- #
def observation_covariance(kappa: np.ndarray, L: int, symbol_var: float = 1.0,
                           noise_var: float = 1.0) -> dict:
    r"""Spectrum of the per-symbol observation covariance: the *only* channel.

    Setup.  With unnormalised ``+-1`` codes (``<c_j,c_j> = L``) and i.i.d.
    Gaussian symbols ``a_j(t) ~ N(0, symbol_var)``, the received vector is

    .. math::

        y_t = \sum_j \sqrt{\kappa_j}\, a_j(t)\, c_j + n_t,
        \qquad n_t \sim N(0, \text{noise\_var}\,I_L).

    Because this is a *linear* combination of independent Gaussians,
    ``y_t | C ~ N(0, Sigma)`` **exactly**, with

    .. math::

        \Sigma = \text{symbol\_var}\,\sum_j \kappa_j c_j c_j^\top
                 + \text{noise\_var}\,I_L .

    Consequences (this is the structural result that replaces the per-chip model)
    ---------------------------------------------------------------------------
    1. The observation law is fully determined by ``Sigma``.  In particular it
       does **not** depend on the *signs* of the interferers' symbols.  So the
       premise of the matched-filter model -- that Eve must resolve the other
       users' symbols in order to attack the code -- is not the premise of the
       *optimal* attack.  Eve can work from second moments alone.
    2. ``(y_1..y_n)`` therefore has ``S_hat = (1/n) sum_t y_t y_t^T`` as a
       **sufficient statistic**, and the codebook enters only through it.
    3. With normalised-ish codes (``<c_j,c_j> = L``, ``<c_i,c_j> ~ 0``) the
       spectrum is::

           noise_var + symbol_var * L * kappa_j   (once per user j)
           noise_var                              (multiplicity L - M)

       so ``Sigma`` reveals the **subspace** ``S = span{c_j}`` and the *powers*
       ``kappa_j``, but **not** which code belongs to which user, and not the
       sign of any code.  With equal ``kappa_j`` even the individual
       ``kappa_j`` are hidden, leaving only the subspace.

    The returned ``snr_eff`` is the relative spectral separation that governs
    how many samples subspace identification needs.  It uses the **smallest**
    ``kappa_j`` (the hardest code to separate out) over the **largest**
    eigenvalue -- the conservative pairing.

    Parameters
    ----------
    kappa : array
        Relative user powers, **including the victim**, i.e. all ``M`` users.
    """
    kappa = np.atleast_1d(np.asarray(kappa, dtype=float))
    kappa = kappa[kappa >= 0]
    if float(noise_var) <= 0:
        raise ValueError("noise_var must be positive")
    shifts = float(symbol_var) * float(L) * kappa          # per-user eigenvalue shift
    lam_gap = float(np.min(shifts)) if shifts.size else 0.0
    lam_noise = float(noise_var)
    lam_max = lam_noise + (float(np.max(shifts)) if shifts.size else 0.0)
    return dict(lambda_signal=lam_gap, lambda_noise=lam_noise,
                lambda_max=lam_max,
                shifts=shifts,
                snr_eff=lam_gap / lam_max if lam_max > 0 else 0.0,
                n_users=int(kappa.size),
                degenerate_signal=bool(np.allclose(kappa, kappa[:1])
                                       if kappa.size else True))


def subspace_identification_samples(L: int, kappa: np.ndarray,
                                   symbol_var: float = 1.0,
                                   noise_var: float = 1.0,
                                   eps: float = 1e-12,
                                   constant: float = 64.0) -> float:
    r"""Samples Eve needs to identify the code **subspace** (rigorous, via
    covariance concentration).

    ``S_hat`` concentrates around ``Sigma`` in operator norm.  For Gaussian
    observations the standard bound (Vershynin, *High-Dimensional Probability*,
    Thm. 4.6.1 / 9.2.4) gives, with probability at least ``1 - eps``,

    .. math::

        \|\hat S - \Sigma\|_2 \;\le\; c\,\|\Sigma\|_2
        \sqrt{\frac{L + \log(2/\epsilon)}{n}} .

    To resolve the code subspace we need this error below half the spectral gap
    ``lambda_signal``, i.e. ``2c * ||Sigma||_2 * sqrt(...) <= lambda_signal``.
    With ``||Sigma||_2 = lambda_max`` and ``snr_eff = lambda_signal/lambda_max``:

    .. math::

        n \;\ge\; \frac{4 c^2 (L + \log(2/\epsilon))}{\text{snr\_eff}^2} .

    The default ``constant = 64 = 4c^2`` corresponds to ``c = 4``, a standard
    non-optimised constant.  **The scaling is the point, not the constant**:
    ``n`` is linear in ``L`` and grows only as ``snr_eff^{-2}``.  Measured
    (``experiments/e11_covariance_attack.py``), ``n`` in the low ``10^4`` already
    suffices for ``L = 1024``, ``M = 8`` at equal powers -- far below this
    conservative bound, so treat the formula as a *sufficient* condition and
    :func:`code_recovery_from_covariance` as the operative measurement.
    """
    spec = observation_covariance(kappa, L, symbol_var, noise_var)
    if spec["snr_eff"] <= 0:
        return float("inf")
    return float(constant * (float(L) + math.log(2.0 / float(eps)))
                 / spec["snr_eff"] ** 2)


def chip_identification_samples(L: int, per_chip_snr: float,
                                target_p_chip: float = 0.99) -> float:
    """Samples the **matched-filter** attack needs to reach ``target_p_chip``.

    This is the inverse of :func:`chip_success_probability`::

        p_chip = Phi(sqrt(per_chip_snr * n / L))
        =>  n = Phi^{-1}(target)^2 * L / per_chip_snr

    ``per_chip_snr`` is the SNR of one chip in one symbol (i.e. the
    ``observation_snr`` argument of :func:`chip_success_probability`).
    """
    from scipy import stats
    if per_chip_snr <= 0:
        return float("inf")
    z = float(stats.norm.isf(1.0 - float(target_p_chip)))
    return float(z ** 2 * float(L) / float(per_chip_snr))


# --------------------------------------------------------------------------- #
# B''. Attack *success probability* at finite n, and the three-way trade-off
# --------------------------------------------------------------------------- #
def subspace_signal_ratio(L: int, kappa: np.ndarray,
                          symbol_var: float = 1.0,
                          noise_var: float = 1.0) -> dict:
    r"""Signal fraction of the code subspace, and its limit as ``L`` grows.

    The Moses--Liu ("signal fraction") statistic of the sample covariance is

    .. math::

        \hat r=\frac{\sum_{i>M}\hat\lambda_i}{\sum_{i\le M}\hat\lambda_i}
        \;\longrightarrow\;
        r(L)=\frac{(L-M)\,n_v}{M\,(n_v+\sigma^2L\bar\kappa)} .

    Small- and large-``L`` limits
    -----------------------------
    At ``L = M`` the numerator vanishes, so ``r(M) = 0``: with exactly as many
    chips as users the subspace fills the whole space and carries no information.
    As ``L -> infinity``,

    .. math::

        r(L)\;\nearrow\;\frac{n_v}{M\,\sigma^2\bar\kappa}\;>\;0 ,

    i.e. ``r`` **increases monotonically toward a positive plateau** (``0.125``
    at ``M=8``, ``n_v=\sigma^2=\bar\kappa=1``).  Two earlier statements in this
    project were wrong and are corrected here:

    * ``r -> 0``: false.  The limit is ``n_v/(M\sigma^2\bar\kappa)``;
    * "a larger ``L`` makes the subspace relatively cleaner": also false.  ``r``
      *rises*, so ``signal_fraction = 1/(1+r)`` *falls* slightly
      (``0.892`` at ``L=256`` to ``0.889`` at ``L=16384``).

    Practical reading: ``r`` is within 1% of its plateau once
    ``L \gtrsim 100\,M\sigma^2\bar\kappa/n_v``, i.e. ``L \gtrsim 800`` in the
    default scenario.  Beyond that, extra chips buy essentially **nothing on this
    axis** -- while still costing a factor ``L`` in symbol rate.  That negative
    result is the honest content of the statistic.

    Returns ``r_exact``, ``r_asymptotic`` (the plateau), ``signal_fraction`` and
    ``saturated`` (``r`` within 1% of the plateau).
    """
    kappa = np.atleast_1d(np.asarray(kappa, dtype=float))
    M = int(kappa.size)
    if M == 0 or L <= 0:
        return dict(r_exact=float("inf"), r_asymptotic=float("inf"),
                    signal_fraction=0.0, saturated=False,
                    noise_eigs=0, signal_eigs=0)
    nv, sv = float(noise_var), float(symbol_var)
    kbar = float(np.mean(kappa))
    lam_sig = nv + sv * float(L) * kbar
    noise_sum = float(L - M) * nv
    signal_sum = float(M) * lam_sig
    r = noise_sum / signal_sum if signal_sum > 0 else float("inf")
    plateau = nv / (M * sv * kbar) if (M * sv * kbar) > 0 else float("inf")
    return dict(r_exact=r, r_asymptotic=plateau,
                signal_fraction=1.0 / (1.0 + r),
                saturated=bool(np.isfinite(plateau) and plateau > 0
                               and r >= 0.99 * plateau),
                noise_eigs=int(L - M), signal_eigs=M,
                lambda_signal=lam_sig)


def subspace_recovery_probability(L: int, kappa: np.ndarray, n_samples: float,
                                  symbol_var: float = 1.0,
                                  noise_var: float = 1.0,
                                  constant: float = 64.0) -> dict:
    r"""Probability that the subspace attack succeeds at a given ``n``.

    There is no honest closed form for this probability -- the finite-sample
    behaviour of the sign iteration is not Gaussian-concentration-friendly.  What
    *is* rigorous is the **sufficient** condition of
    :func:`subspace_identification_samples`:

    .. math::

        n \;\ge\; n_{\text{th}}(L)=\frac{4c^2(L+\log(2/\epsilon))}{\text{snr\_eff}^2}
        \quad\Longrightarrow\quad \|\hat S-\Sigma\|_2\le\tfrac12\lambda_{\text{signal}},

    which implies the subspace is identified (Davis--Kahan) and hence the codes
    are recoverable up to the ``\pm1`` search.

    We therefore report a **step**, not a smooth curve::

        p_recover(n) = 1  if n >= n_th(L),  else 0

    and separately the *margin* ``n / n_th``.  This is deliberate: quoting a
    logistic fit would invent a precision we do not have.  The measured
    transition (``experiments/e12_refresh_period.py``) sits about two orders of
    magnitude **below** ``n_th``, i.e. the bound is sufficient but very loose;
    ``conservative_ratio`` records that gap.

    Also returned: ``snr_eff``, ``lambda_signal``, ``lambda_max``,
    ``r_exact`` from :func:`subspace_signal_ratio`.
    """
    spec = observation_covariance(kappa, L, symbol_var, noise_var)
    n_th = subspace_identification_samples(L, kappa, symbol_var, noise_var,
                                           constant=constant)
    ratio = subspace_signal_ratio(L, kappa, symbol_var, noise_var)
    ok = bool(np.isfinite(n_th) and float(n_samples) >= n_th)
    return dict(n_samples=float(n_samples), n_threshold=float(n_th),
                p_recover=1.0 if ok else 0.0,
                margin=float(n_samples) / n_th if np.isfinite(n_th) and n_th > 0
                else float("inf"),
                snr_eff=spec["snr_eff"], lambda_signal=spec["lambda_signal"],
                lambda_max=spec["lambda_max"], r_exact=ratio["r_exact"],
                signal_fraction=ratio["signal_fraction"], recovered=ok)


def three_way_tradeoff(L: int, kappa, refresh_periods,
                       block_symbols: float = 1e8, chip_rate: float = 1e9,
                       symbol_var: float = 1.0, noise_var: float = 1.0,
                       eps: float = 1e-12, constant: float = 64.0,
                       n_interferers: int | None = None) -> dict:
    r"""The three costs as explicit functions of the **PRF refresh period** ``T``.

    ``T`` = how many symbols pass before the PRF re-draws the codebook from the
    seed.  It is the one knob that trades all three objectives, so the whole
    analysis is written as a function of it.

    (a) Security: samples per codebook
    ---------------------------------
    Eve can only average over the symbols that share **one** code.  So the
    sample count she gets is ``n = min(T, block)``, and the attack succeeds when
    that clears the subspace-identification threshold

    .. math::

        n_{\text{th}}(L)=\frac{4c^2\bigl(L+\log(2/\epsilon)\bigr)}
                              {\text{snr\_eff}(L)^2} .

    Since ``n_{\text{th}}`` is **independent of T**, the condition is a clean
    threshold in the refresh period itself:

    .. math::

        T \;<\; n_{\text{th}}(L)\quad\Longrightarrow\quad\text{attack fails}.

    This is the honest statement of the security leg: it is a step, and the
    margin is ``T / n_th``.  The measured transition sits far below ``n_th``
    (the concentration constant ``4c^2`` is loose), so ``T`` must be *much*
    smaller than the quoted value -- never read ``n_th`` as the safe ``T``.

    (b) MUI: fluctuation shrinks as ``1/sqrt(F)``, ``F = block/T``
    -------------------------------------------------------------
    .. math::

        \text{rel.std}[\xi_{\text{block}}]
        =\sqrt{\frac{2\sum_j\kappa_j^2}{(\sum_j\kappa_j)^2}}\cdot\frac{1}{\sqrt F}
        =\sqrt{\frac{2\sum_j\kappa_j^2}{(\sum_j\kappa_j)^2}}\sqrt{\frac{T}{B}} .

    **This is the key identity**: refining the refresh (smaller ``T``, larger
    ``F``) is the *only* way to stabilise the MUI.  The mean
    ``(1/L)\sum_j\kappa_j`` does not change at all.  So on the MUI axis a small
    ``T`` is strictly good.

    (c) Compute: PRF throughput ``M L / T`` bits per symbol
    ------------------------------------------------------
    .. math::

        \text{PRF}(T)=\frac{M\,L}{T}\ \text{[bits/symbol]},
        \qquad \text{rate}=\frac{M\,L\,R_c/L}{T}=\frac{M R_c}{T}\ \text{[bit/s]} .

    So PRF cost is ``1/T``: a small ``T`` is strictly bad.

    Conflict
    --------
    ``T`` is therefore pushed **down** by MUI stability and **up** by PRF cost,
    and the security constraint only requires ``T < n_{\text{th}}``.  There is no
    unique optimum without a price on PRF throughput; what the analysis gives is
    the feasible interval

    .. math::

        T \;\le\; \min\bigl(n_{\text{th}}(L),\; T_{\text{PRF budget}}\bigr),

    where the second term is set by whatever PRF throughput hardware allows.

    Returns
    -------
    dict of arrays keyed by ``T``, plus the per-``T`` table under ``rows``.
    Each row: ``T``, ``n_samples_per_codebook``, ``n_threshold``,
    ``p_recover``, ``margin``, ``n_frames``, ``xi_mean``, ``xi_rel_std``,
    ``xi_eps``, ``prf_bits_per_symbol``, ``prf_bit_per_s``,
    ``eve_flops_per_block``, ``bob_flops_per_block``.
    """
    Ts = np.atleast_1d(np.asarray(refresh_periods, dtype=float))
    kappa = np.atleast_1d(np.asarray(kappa, dtype=float))
    M = int(kappa.size)
    L = int(L)
    # The MUI leg sees interferers only; the attack leg sees everybody.
    K = int(n_interferers) if n_interferers is not None else max(M - 1, 1)
    kappa_mui = kappa[:K]
    spec = observation_covariance(kappa, L, symbol_var, noise_var)
    n_th = subspace_identification_samples(L, kappa, symbol_var, noise_var,
                                           eps=eps, constant=constant)
    xi_mean = float(np.sum(kappa_mui) / L)
    rel_std1 = float(np.sqrt(2.0 * np.sum(kappa_mui ** 2))
                     / np.sum(kappa_mui)) if np.sum(kappa_mui) > 0 else 0.0
    symbol_rate = chip_rate / float(L)

    out = dict(T=Ts, rows=[], n_interferers=K)
    for T in Ts:
        T = max(float(T), 1.0)
        n_sym = float(min(T, block_symbols))
        F = max(block_symbols / T, 1.0)
        ok = bool(np.isfinite(n_th) and n_sym < n_th)
        from scipy import stats
        if np.allclose(kappa_mui, kappa_mui[:1]):
            dof = int(round(F)) * K
            xi_eps = float(kappa_mui[0]
                           * float(stats.chi2.ppf(1.0 - 1e-12, dof))
                           / (L * float(F)))
        else:
            var = 2.0 * np.sum(kappa_mui ** 2) / (L ** 2 * F)
            a = xi_mean ** 2 / var if var > 0 else np.inf
            xi_eps = (float(stats.gamma.ppf(1.0 - 1e-12, a=a,
                                            scale=var / xi_mean))
                      if np.isfinite(a) else xi_mean)
        out["rows"].append(dict(
            T=T,
            n_samples_per_codebook=n_sym,
            n_threshold=float(n_th),
            p_recover=0.0 if ok else 1.0,
            attack_blocked=ok,
            margin=n_sym / n_th if np.isfinite(n_th) and n_th > 0 else np.inf,
            n_frames=float(F),
            xi_mean=xi_mean,
            xi_rel_std=rel_std1 / math.sqrt(F),
            xi_eps=xi_eps,
            prf_bits_per_symbol=float(M) * L / T,
            prf_bit_per_s=float(M) * L / T * symbol_rate,
            refresh_rate_hz=1.0 / T * symbol_rate,
            eve_flops_per_block=float(L) ** 2 * float(block_symbols),
            bob_flops_per_block=float(M) * L * float(block_symbols),
        ))
    for k in (out["rows"][0] if out["rows"] else {}):
        out[k] = np.asarray([r[k] for r in out["rows"]])
    out["L"] = L
    out["M"] = M
    out["n_threshold"] = float(n_th)
    out["snr_eff"] = spec["snr_eff"]
    out["xi_mean"] = xi_mean
    out["xi_rel_std_frozen"] = rel_std1
    return out


def blind_attack_envelope(L: int, kappa: np.ndarray, per_chip_snr: float,
                          symbol_var: float = 1.0, noise_var: float = 1.0,
                          eps: float = 1e-12,
                          target_p_chip: float = 0.99) -> dict:
    r"""Both blind attacks side by side, and the **envelope** (the cheaper one).

    The project's security accounting used only the matched filter, whose cost
    is ``~ z^2 L / snr_chip``.  The subspace attack costs
    ``~ 4c^2 (L + log(2/eps)) / snr_eff^2``.

    What the numbers actually say (``experiments/e11_covariance_attack.py``)
    -----------------------------------------------------------------------
    If ``per_chip_snr`` is tied to the same physical power as ``snr_eff`` (as it
    is in E11), then ``snr_eff ~ kappa L /(1 + kappa L)`` and the **matched
    filter is cheaper in sample count at essentially every power**; the subspace
    cost grows as ``snr_eff^{-2}`` against the filter's ``snr_chip^{-1}``.  So
    "the subspace attack is the cheaper attack" is **not** the claim this
    function supports, and the envelope usually returns the chip cost.

    Why the subspace attack is nevertheless the relevant threat
    ----------------------------------------------------------
    Cost is not the whole story; **premise** is.  The matched filter presumes
    Eve can resolve the *other users' symbols* well enough to form per-chip
    decisions on the target code.  But the observation law is

        y_t | C ~ N(0, Sigma),

    which is blind to those symbol signs (see
    :func:`observation_covariance`).  So the per-chip model answers a question
    Eve never has to ask, and it returns ``p_chip -> 1/2`` ("no leakage") in
    exactly the weak-signal regime where a second-moment attack still works:
    :func:`code_recovery_from_covariance` recovers every code at ``n = 10^4``
    symbols even for a 0.01-power interferer.  The envelope is reported for
    completeness, not as the ground for the security conclusion.

    See ``docs/08_open_problems.md`` §1.1 and ``docs/09_model_vs_bound.md`` §5.
    """
    spec = observation_covariance(kappa, L, symbol_var, noise_var)
    n_chip = chip_identification_samples(L, per_chip_snr, target_p_chip)
    n_sub = subspace_identification_samples(L, kappa, symbol_var, noise_var, eps)
    n_env = min(n_chip, n_sub)
    return dict(n_chip=n_chip, n_subspace=n_sub, n_samples=n_env,
                chip_is_cheaper=bool(n_chip <= n_sub),
                ratio_subspace_over_chip=(n_sub / n_chip
                                          if n_chip > 0 and np.isfinite(n_chip)
                                          else np.nan),
                snr_eff=spec["snr_eff"], lambda_signal=spec["lambda_signal"],
                lambda_noise=spec["lambda_noise"])


def code_ambiguity_bits(n_users: int, equal_powers: bool = False) -> float:
    r"""Bits of the codebook that a **perfect** covariance leaves ambiguous.

    This is NOT the amount of work the attack needs -- see
    :func:`blind_attack_envelope` for the sample cost.  It is the residual
    ambiguity *after* the subspace is known, and it is what
    :func:`count_pm1_vectors_in_span` measures.

    The exact covariance pins down ``range(Sigma) = span{c_j}`` exactly (the
    noise only shifts the spectrum).  What is left is: how many ``+-1`` vectors
    lie in that subspace, and can Eve tell which is which?

    * A random rank-``M`` subspace spanned by ``+-1`` vectors contains **exactly
      the ``2M`` vectors** ``{+-c_1, ..., +-c_M}`` with high probability (the
      odd extra vector appears occasionally at small ``L``; the tests at
      ``L in {16, 18}`` show counts of ``2M`` and rarely ``2M + 2``).  So the
      ``+-1`` prior **completely breaks** the continuous ambiguity that would
      otherwise make blind separation impossible.
    * A global sign per code is irrelevant for despreading (``<+-c_0, y>`` gives
      the same symbol up to sign).
    * If the ``kappa_j`` are **distinct**, the eigenvectors of ``Sigma`` order
      the codes by power, so the ordering is known: ambiguity ``0`` bits.
    * If the ``kappa_j`` are **equal**, the non-noise eigenvalues are equal and
      only the *set* of codes is known, not which user owns which: ambiguity
      ``log2(M!)`` ("labelling").  That is 15.3 bit for ``M = 8``.
    """
    M = max(int(n_users), 1)
    if not equal_powers:
        return 0.0
    return float(math.lgamma(M + 1) / math.log(2.0))


def count_pm1_vectors_in_span(codebook: np.ndarray) -> int:
    r"""Number of ``+-1`` vectors lying in ``span(codebook)`` (brute force).

    Only usable for small ``L`` (``2^L`` enumeration).  This is the measurement
    behind :func:`code_ambiguity_bits`: for a random ``M``-dimensional subspace
    spanned by ``+-1`` codes the count is ``2M`` (each code up to sign), so
    knowing the subspace hands Eve the codes outright once she enumerates the
    ``+-1`` lattice points inside it.
    """
    C = np.atleast_2d(np.asarray(codebook, dtype=float))
    L = C.shape[1]
    if L > 20:
        raise ValueError("brute force is only sensible for L <= 20")
    Q, _ = np.linalg.qr(C.T)
    Q = Q[:, :C.shape[0]]
    count = 0
    for mask in range(2 ** L):
        v = np.array([1.0 if (mask >> i) & 1 else -1.0 for i in range(L)])
        if np.allclose(Q @ (Q.T @ v), v, atol=1e-9):
            count += 1
    return count


def recover_codes_from_subspace(subspace_basis: np.ndarray,
                                iters: int = 2000, restarts: int = 200,
                                tol: float = 0.05, seed: int = 1) -> dict:
    r"""Alternating projections: find the ``+-1`` vectors inside a subspace.

    Given an orthonormal basis ``Q`` (``L x M``) of ``span{c_j}``, find
    ``x in {+-1}^L`` with ``Q Q^T x = x`` by the iteration
    ``x <- sign(Q Q^T x)`` from random starts.

    This is the step that converts "Eve knows the code subspace" into "Eve knows
    the codes".  Without the ``+-1`` prior the subspace would be useless -- any
    orthogonal rotation of the codes spans the same space and gives statistically
    identical Gaussian observations (the classical ICA unidentifiability).  The
    ``+-1`` constraint is discrete and is *not* preserved by rotations, which is
    exactly why the prior is decisive.

    Acceptance.  At finite ``n`` the true codes are **not** exactly in the
    estimated subspace, so an absolute residual test is wrong (it would reject
    every candidate).  A candidate is accepted when (i) the sign iteration
    reaches a **fixed point** and (ii) its per-chip projection RMS
    ``||QQ^T x - x||_2 / sqrt(L)`` is at most ``tol``.  ``tol`` is therefore in
    units of one code chip (``|c_ij| = 1``), not an absolute distance.
    ``exact=True`` in the result means the residual was at float noise level.

    Returns
    -------
    dict
        ``vectors``  ``(k, L)`` distinct ``+-1`` vectors found (typically
        ``2M``: each code up to a global sign, which is irrelevant for
        despreading).
        ``projection_rms``  per-vector RMS residual (same order as ``vectors``).
        ``exact``  whether every accepted vector lies in the subspace to float
        precision.
        ``n_restarts``  number of random starts used.
    """
    Q = np.atleast_2d(np.asarray(subspace_basis, dtype=float))
    L = Q.shape[0]
    rng = np.random.default_rng(seed)
    found: dict[tuple, np.ndarray] = {}
    rms: dict[tuple, float] = {}
    for _ in range(int(restarts)):
        x = rng.choice([-1.0, 1.0], size=L)
        converged = False
        # Cycle detection.  The sign iteration is `x <- sign(QQ^T x)`; when the
        # subspace contains no `+-1` vector -- the common case *during a failed
        # attack* -- it settles into a short limit cycle, typically of length 2.
        # Without a guard it would burn the whole `iters` budget.
        #
        # Implementation note: we keep the last few states as plain arrays and
        # compare with `np.array_equal` (a C loop).  An earlier version kept a
        # Python `set` of 1024-element tuples, which hashed ~1 KB per iteration
        # and dominated the runtime of every failed attack.
        history: list[np.ndarray] = []
        max_cycle = 4
        for _ in range(int(iters)):
            xn = np.sign(Q @ (Q.T @ x))
            xn[xn == 0] = 1.0                      # resolve exact-zero ties
            if np.array_equal(xn, x):
                converged = True
                break
            if any(np.array_equal(xn, h) for h in history):
                break                              # limit cycle, give up
            history.append(xn)
            if len(history) > max_cycle:
                del history[0]
            x = xn
        if not converged:
            continue
        r = float(np.linalg.norm(Q @ (Q.T @ x) - x) / math.sqrt(L))
        if r <= tol:
            key = tuple(x.astype(np.int8))
            found[key] = x
            rms[key] = r
    keys = list(found)
    return dict(vectors=(np.array([found[k] for k in keys]) if keys
                         else np.zeros((0, L))),
                projection_rms=np.array([rms[k] for k in keys]),
                exact=bool(all(rms[k] <= 1e-9 for k in keys)),
                n_restarts=int(restarts))


def coded_covariance(codebook: np.ndarray, kappa: np.ndarray,
                     symbol_var: float = 1.0, noise_var: float = 1.0) -> np.ndarray:
    r"""The exact ``Sigma`` of :func:`observation_covariance`, as a matrix.

    ``Sigma = symbol_var * sum_j kappa_j c_j c_j^T + noise_var * I``.
    Kept separate so tests can compare the eigen-spectrum against the closed
    form returned by :func:`observation_covariance`.
    """
    C = np.atleast_2d(np.asarray(codebook, dtype=float))
    kap = np.atleast_1d(np.asarray(kappa, dtype=float))
    if kap.size != C.shape[0]:
        raise ValueError("kappa must have one entry per codebook row")
    L = C.shape[1]
    return float(symbol_var) * ((C.T * kap) @ C) + float(noise_var) * np.eye(L)


def code_recovery_from_covariance(codebook: np.ndarray, kappa: np.ndarray,
                                  n_samples: int, symbol_var: float = 1.0,
                                  noise_var: float = 1.0, chunk: int = 25000,
                                  restarts: int = 200, iters: int = 2000,
                                  tol: float = 0.05, seed: int = 1) -> dict:
    r"""Simulate the **subspace attack** end to end, and measure what Eve recovers.

    This is the measurement that turns the covariance channel from a *model* into
    an observed fact.  Steps:

    1. Sample ``n_samples`` observations ``y_t = A z_t`` in chunks, where
       ``A A^T = Sigma`` (Cholesky).  Chunking is essential: materialising
       ``(n_samples, L)`` at ``n = 10^8`` would need ~763 GiB.
    2. ``S_hat = (1/n) sum_t y_t y_t^T`` (accumulated, never stored).
    3. Take the top-``M`` eigenvectors as the estimated code subspace ``Q``.
    4. Enumerate ``+-1`` vectors in ``Q`` via
       :func:`recover_codes_from_subspace`.
    5. Score each true code by its best ``|cos|`` against the found vectors.
       A code counts as recovered when that best correlation is ``>= 0.9``;
       around 1/``sqrt(L)`` (~0.03 here) means not recovered.

    Crucially step 3 alone recovers **nothing** (the eigenvectors are arbitrary
    orthonormal mixtures of the codes); the recovery comes from the discrete
    ``+-1`` prior in step 4.  Both numbers are returned so the distinction is
    visible.

    Parameters
    ----------
    codebook : array ``(M, L)``
        The true ``+-1`` codes.  In the protocol these are PRF outputs; the
        attack assumes only that Eve knows ``+-1``, not their values.
    kappa : array ``(M,)``
        Received powers, victim included.

    Returns
    -------
    dict
        ``recovery_fraction``  fraction of codes recovered (``>= 0.9`` corr).
        ``per_code_best_corr``  best ``|cos|`` per true code.
        ``n_found``  number of distinct ``+-1`` vectors found in the subspace.
        ``err_operator_norm``  ``||S_hat - Sigma||_2`` (concentration check).
        ``eigvec_best_corr_worst/mean``  what the *eigenvectors alone* achieve.
        ``n_samples``, ``snr_eff``, ``n_theory``  the cost accounting.
    """
    C = np.atleast_2d(np.asarray(codebook, dtype=float))
    kap = np.atleast_1d(np.asarray(kappa, dtype=float))
    M, L = C.shape
    n = int(n_samples)
    if n <= 0:
        raise ValueError("n_samples must be positive")

    Sigma = coded_covariance(C, kap, symbol_var, noise_var)
    A = np.linalg.cholesky(Sigma)
    rng = np.random.default_rng(seed)

    gram = np.zeros((L, L))
    done = 0
    while done < n:
        m = min(int(chunk), n - done)
        Z = rng.standard_normal((m, L))
        gram += Z.T @ Z
        done += m
    S = A @ (gram / n) @ A.T

    lam, U = np.linalg.eigh(S)
    Q = U[:, -M:]

    Cn = C / np.linalg.norm(C, axis=1, keepdims=True)
    Qn = Q / np.linalg.norm(Q, axis=0, keepdims=True)
    eigcorr = np.abs(Qn.T @ Cn.T).max(axis=0)

    rec = recover_codes_from_subspace(Q, iters=iters, restarts=restarts,
                                      tol=tol, seed=seed)
    found = rec["vectors"]
    if found.size:
        Fn = found / np.linalg.norm(found, axis=1, keepdims=True)
        best = np.abs(Fn @ Cn.T).max(axis=0)
    else:
        best = np.zeros(M)

    spec = observation_covariance(kap, L, symbol_var, noise_var)
    return dict(
        n_samples=n,
        recovery_fraction=float(np.mean(best >= 0.9)),
        per_code_best_corr=[float(x) for x in best],
        n_found=int(found.shape[0]),
        n_exact=int(sum(1 for r in rec["projection_rms"] if r <= 1e-9)),
        worst_projection_rms=(float(rec["projection_rms"].max())
                              if rec["projection_rms"].size else None),
        err_operator_norm=float(np.linalg.norm(S - Sigma, 2)),
        lambda_top=float(lam[-1]),
        lambda_noise_top=float(lam[L - M - 1]) if L - M - 1 >= 0 else 0.0,
        eigvec_best_corr_worst=float(eigcorr.min()),
        eigvec_best_corr_mean=float(eigcorr.mean()),
        snr_eff=spec["snr_eff"],
        n_theory=subspace_identification_samples(L, kap, symbol_var, noise_var),
    )


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
    r"""Rigorous refresh constraint: binomial **tail** *plus* the entropy term.

    What the previous version got right and what it missed
    ------------------------------------------------------
    It already used an ``eps_code`` target rather than the degenerate
    ``H_min > 0`` test, and it already used the binomial tail rather than the
    mean of ``k_known``.  What it missed is that the tail event and the
    guessing probability are **two different things**:

    * ``P(k_known >= k_max)`` is the probability that Eve exceeds her chip
      budget.  Conditioning on that event says nothing about how *likely* she
      is to then guess the seed -- and if she has ``k_max`` chips she has
      consumed the entire security margin, so the conditional guessing
      probability is essentially 1.
    * ``2^{-(seed - k)}`` is the guessing probability **given** ``k`` chips.

    A sound statement must add them::

        eps_code(f) <= P(k > k*) + 2^{-(seed - k*)}       for every k*
        so choose k* to minimise this (the two terms trade off one chip at a
        time: the first falls, the second doubles).

    Only using the first term silently sets the second to zero, which is exactly
    what makes ``f_min`` land at ``1 - k_max/L``; only using the second term
    (the mean-based route) silently sets the first to zero.  Both are wrong in
    the *optimistic* direction.  This function does the union bound properly.

    The resulting ``f_min`` is slightly **larger** than the old
    ``1 - (k_max)/L`` because the optimum ``k*`` is a little below ``k_max``.

    Returns a dict including ``k_star`` (the optimised chip count),
    ``p_exceed`` and ``p_guess`` so the split between the two terms is visible.
    """
    from scipy import stats

    if L <= 0:
        raise ValueError("L must be positive")
    if not 0.0 < eps_code <= 1.0:
        raise ValueError("eps_code must be in (0, 1]")
    k_req = required_entropy_bits(eps_code)
    k_max = seed_bits - k_req          # chips that would consume all the margin
    p = float(np.clip(p_blind, 0.0, 1.0))
    base = dict(seed_bits=int(seed_bits), L=int(L), p_blind=p,
                eps_code=eps_code, k_req=k_req, k_max_allowed=float(k_max))

    if seed_bits <= k_req:
        return {**base, "f_min": 1.0, "feasible": False, "violation": 1.0,
                "note": "seed too short to hold the security margin"}

    if p <= 0.0:
        return {**base, "f_min": 0.0, "feasible": True, "violation": 0.0,
                "k_star": 0, "note": "p_blind = 0: no leakage"}

    def eps_of(f: float, k_star: int | None = None) -> tuple[float, int, float, float]:
        """(eps_code, k*, P(k>k*), 2^{-(seed-k*)}) for a refresh fraction f."""
        n_reused = int(round((1.0 - f) * L))
        # A chip that is not refreshed carries over from the previous code; the
        # leaked chips are therefore capped by min(n_reused, k_star).
        k_hi = min(int(k_star if k_star is not None else n_reused), n_reused)
        lo_k, best = 0, None
        # optimise k* over the reachable range (the sum is convex: the binomial
        # tail falls with k*, the guessing term doubles)
        for k in range(lo_k, k_hi + 1):
            p_exc = (0.0 if n_reused <= k
                     else (1.0 if p >= 1.0
                           else float(stats.binom.sf(k, n_reused, p))))
            p_gue = float(min(1.0, 2.0 ** (-(seed_bits - k))))
            tot = p_exc + p_gue
            if best is None or tot < best[0]:
                best = (tot, k, p_exc, p_gue)
        assert best is not None
        return best

    e1 = eps_of(1.0)
    if e1[0] > eps_code:
        return {**base, "f_min": 1.0, "feasible": False, "violation": e1[0],
                "k_star": e1[1], "p_exceed": e1[2], "p_guess": e1[3],
                "note": "no refresh rate satisfies the bound; even f = 1 loses "
                        "the seed (2^{-seed} exceeds the target)"}
    e0 = eps_of(0.0)
    if e0[0] <= eps_code:
        return {**base, "f_min": 0.0, "feasible": True, "violation": e0[0],
                "k_star": e0[1], "p_exceed": e0[2], "p_guess": e0[3],
                "binds": False,
                "note": "bound already met without refreshing"}

    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if eps_of(mid)[0] > eps_code:
            lo = mid
        else:
            hi = mid
    e = eps_of(hi)
    return {**base, "f_min": float(hi), "feasible": True, "violation": e[0],
            "k_star": e[1], "p_exceed": e[2], "p_guess": e[3], "binds": True,
            "f_min_legacy": float(np.clip(
                1.0 - (k_max) / (L * p) if L * p > 0 else 1.0, 0.0, 1.0)),
            "f_min_mean": float(refresh_security_constraint(
                seed_bits, L, p, eps_code)["f_min"]),
            "note": ("rigorous: eps_code <= P(k>k*) + 2^{-(seed-k*)}, k* "
                     "optimised per refresh fraction")}


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
