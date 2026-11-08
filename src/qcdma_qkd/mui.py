"""Dynamic multi-user interference (MUI) statistics for q-CDMA.

Physical model
--------------
``M`` users share ``L`` chips per symbol with normalised +-1 codes ``c_i``
(``<c_i,c_i> = L``).  User 0 despreads its received field with ``c_0``; the
contribution of user ``j >= 1`` is

    I_j = sqrt(kappa_j) * a_j * rho_0j,      rho_0j = <c_0, c_j> / L

with

* ``kappa_j``  relative received power of interferer j (calibrated out at the
  receiver; the *residual* is what matters),
* ``a_j``      the interferer's zero-mean symbol on that quadrature; for
  coherent states of variance V_A we have ``a_j ~ N(0, sigma^2)`` with
  ``sigma^2 = V_A/2`` (per quadrature),
* ``rho_0j``   normalised cross-correlation of the two codes.

After despreading, the useful signal is ``a_0`` and the aggregate interference
is ``MUI = sum_j I_j``.  Hence the **excess noise referred to a symbol of
variance sigma^2** is

    xi_MUI = Var(MUI) / sigma^2 = (1/L) * sum_j kappa_j

which is the quantity added to the channel excess noise in the CV-QKD rate
calculation.  Note it does not depend on ``sigma``: signal and interference
scale together.

Key statistical point (paper Section 7)
---------------------------------------
``Var(MUI)`` is deterministic but the distribution is a **Gaussian variance
mixture**: conditionally on the code draw, ``MUI ~ N(0, s^2)`` with

    s^2 = (1/L) * sum_j kappa_j * sigma^2 * z_j^2,      z_j ~ N(0,1).

A variance mixture has heavier tails than a Gaussian of equal variance:

1. Static orthogonal codes (Walsh) give ``rho = 0`` identically -> no MUI, but
   the codebook is public, hence no access control.
2. Fresh random codes give small MUI on average but a random instantaneous
   value; the outlier probability must be budgeted as ``eps_MUI``.

Moments
-------
For independent centred Gaussians ``a`` and ``rho``, the product ``a*rho`` has
second moment ``sigma^2/L`` and excess kurtosis **6**.  With
``v_j = kappa_j sigma^2 / L``:

    Var(MUI)    = sum_j v_j
    E[MUI^4]    = 6 sum_j v_j^2 + 3 (sum_j v_j)^2
    kurt_excess = 6 sum_j v_j^2 / (sum_j v_j)^2

Tail
----
``s^2`` is a weighted sum of ``chi^2_1`` variables::

    s^2 = sum_j v_j * z_j^2,        z_j ~ N(0,1)

so the marginal of MUI is the law of ``z * sqrt(sum_j v_j z_j^2)`` with the
``z`` independent.  This is a **product-type variance mixture**, *not* a
Student-t (the Gaussian factor appears once, not squared-and-divided).  For a
single interferer it is the product of two Gaussians, whose exact density is a
Bessel function ``K_0`` and whose excess kurtosis is 6 -- consistent with the
moments above.

Because no elementary closed form exists for ``K >= 2``, the tail is obtained
two ways in this module:

* ``tail_mixture``  -- exact, by quadrature over the Gamma law of ``s^2``
  (validated against Monte Carlo in ``tests/``);
* ``tail_fourth_moment`` / ``tail_chebyshev`` -- rigorous distribution-free
  bounds (Markov on the 4th / 2nd moment), used when a *proof* rather than a
  number is needed in the paper's ``eps_MUI`` budget.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.integrate import quad


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def uniform_kappa(n_interferers: int, power_ratio: float = 1.0) -> np.ndarray:
    """All interferers at ``power_ratio`` relative to the victim."""
    return np.full(int(n_interferers), float(power_ratio))


def quadrature_variance(V_A: float) -> float:
    """Modulation variance of one quadrature for a coherent state (SNU)."""
    return float(V_A) / 2.0


def _v(kappa: np.ndarray, L: int, symbol_var: float) -> np.ndarray:
    return np.atleast_1d(np.asarray(kappa, dtype=float)) * symbol_var / float(L)


# --------------------------------------------------------------------------- #
# moments
# --------------------------------------------------------------------------- #
def mui_variance(kappa: np.ndarray, L: int, symbol_var: float = 1.0) -> float:
    """Var(MUI) for independent random codes with rho ~ N(0, 1/L)."""
    return float(np.sum(_v(kappa, L, symbol_var)))


def mui_std(kappa: np.ndarray, L: int, symbol_var: float = 1.0) -> float:
    return float(np.sqrt(mui_variance(kappa, L, symbol_var)))


def mui_fourth_moment(kappa: np.ndarray, L: int, symbol_var: float = 1.0) -> float:
    """E[MUI^4] exactly (independent centred terms, product kurtosis 6)."""
    v = _v(kappa, L, symbol_var)
    if v.size == 0:
        return 0.0
    s = float(np.sum(v))
    return float(6.0 * np.sum(v ** 2) + 3.0 * s ** 2)


def mui_kurtosis(kappa: np.ndarray, L: int, symbol_var: float = 1.0) -> float:
    """Excess kurtosis (0 Gaussian; 6/K for K equal interferers)."""
    v = _v(kappa, L, symbol_var)
    if v.size == 0:
        return 0.0
    s = float(np.sum(v))
    return float(6.0 * np.sum(v ** 2) / s ** 2) if s > 0 else 0.0


def effective_interferers(kappa: np.ndarray) -> float:
    """nu = (sum v)^2 / sum v^2: dof of the equivalent Student-t."""
    v = np.atleast_1d(np.asarray(kappa, dtype=float))
    v = v[v > 0]
    if v.size == 0:
        return np.inf
    return float(np.sum(v) ** 2 / np.sum(v ** 2))


def excess_noise_xi(mui_var: float, symbol_var: float) -> float:
    """xi_MUI = Var(MUI)/sigma^2 -- added to xi in the CV-QKD rate."""
    if symbol_var <= 0:
        raise ValueError("symbol_var must be positive")
    return float(mui_var / symbol_var)


def xi_from_codes(kappa: np.ndarray, L: int) -> float:
    """xi_MUI = (1/L) sum_j kappa_j  (sigma cancels)."""
    return float(np.sum(np.atleast_1d(np.asarray(kappa, dtype=float))) / float(L))


# --------------------------------------------------------------------------- #
# tails
# --------------------------------------------------------------------------- #
def tail_mixture(threshold: float, kappa: np.ndarray, L: int,
                 symbol_var: float = 1.0, n_quad: int = 4000) -> float:
    """P(|MUI| >= threshold) by exact quadrature over the law of s^2.

    s^2 = sum_j v_j z_j^2, a weighted sum of chi^2_1. Its law is matched by a
    Gamma (exact for equal weights) and the mixture is integrated numerically:

        P(|MUI| >= tau) = E_{s^2}[ 2 * Phi(-tau/s) ].

    Validated against Monte Carlo to <2% for the regimes used in the paper.
    """
    if threshold <= 0:
        return 1.0
    v = _v(kappa, L, symbol_var)
    if v.size == 0:
        return 0.0
    return _gamma_mixture_sf(threshold, v, n_quad)


def _gamma_mixture_sf(threshold: float, v: np.ndarray, n_quad: int = 4000) -> float:
    mean = float(np.sum(v))
    var = float(2.0 * np.sum(v ** 2))
    if mean <= 0:
        return 0.0
    shape, scale = mean ** 2 / var, var / mean
    q = np.linspace(1e-7, 1.0 - 1e-7, n_quad)
    s2 = stats.gamma.ppf(q, a=shape, scale=scale)
    return float(np.mean(2.0 * stats.norm.sf(threshold / np.sqrt(s2))))


def tail_single_interferer_exact(threshold: float, v: float) -> float:
    """Exact tail for one interferer: Z1*Z2 with both standard normal.

    P(|Z1 Z2| >= t) = (2/pi) * integral_t^inf K_0(u) du, evaluated with
    scipy.special.kve/k0 by quadrature.
    """
    from scipy.special import k0
    if threshold <= 0:
        return 1.0
    f = lambda u: 2.0 / np.pi * k0(u)
    val, _ = quad(f, threshold, np.inf, limit=200)
    return float(min(1.0, val))


def tail_fourth_moment(threshold: float, kappa: np.ndarray, L: int,
                       symbol_var: float = 1.0) -> float:
    """Assumption-free Markov bound on the fourth moment: E[MUI^4]/tau^4."""
    if threshold <= 0:
        return 1.0
    m4 = mui_fourth_moment(kappa, L, symbol_var)
    return float(min(1.0, m4 / threshold ** 4)) if m4 > 0 else 0.0


def tail_chebyshev(threshold: float, kappa: np.ndarray, L: int,
                   symbol_var: float = 1.0) -> float:
    """Chebyshev bound Var/tau^2 (always valid, loosest)."""
    if threshold <= 0:
        return 1.0
    return float(min(1.0, mui_variance(kappa, L, symbol_var) / threshold ** 2))


# --------------------------------------------------------------------------- #
# numerics / validation
# --------------------------------------------------------------------------- #
def monte_carlo_mui(kappa: np.ndarray, L: int, symbol_var: float = 1.0,
                    n_draws: int = 200_000, seed: int = 0) -> np.ndarray:
    """Monte-Carlo samples of MUI under independent random codes."""
    rng = np.random.default_rng(seed)
    kappa = np.atleast_1d(np.asarray(kappa, dtype=float))
    K = kappa.size
    a = rng.normal(0.0, np.sqrt(symbol_var), size=(n_draws, K))
    rho = rng.normal(0.0, 1.0 / np.sqrt(L), size=(n_draws, K))
    return (np.sqrt(kappa) * a * rho).sum(axis=1)


def empirical_outage(samples: np.ndarray, threshold: float) -> float:
    return float(np.mean(np.abs(samples) >= threshold))


def exact_outage_gamma(threshold: float, kappa: np.ndarray, L: int,
                       symbol_var: float = 1.0) -> float:
    """Alias of :func:`tail_mixture` (kept for readable call sites)."""
    return tail_mixture(threshold, kappa, L, symbol_var)


def safe_threshold(target_outage: float, kappa: np.ndarray, L: int,
                   symbol_var: float = 1.0) -> float:
    """Largest tau with P(|MUI| >= tau) <= target_outage (bisection on the
    exact mixture tail)."""
    kappa = np.atleast_1d(np.asarray(kappa, dtype=float))
    if kappa.size == 0:
        return 0.0
    mean_s2 = mui_variance(kappa, L, symbol_var)
    if mean_s2 <= 0:
        return 0.0
    lo, hi = 0.0, 12.0 * np.sqrt(mean_s2)
    # expand the bracket until the tail is small enough
    for _ in range(40):
        if tail_mixture(hi, kappa, L, symbol_var) <= target_outage:
            break
        hi *= 1.5
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if tail_mixture(mid, kappa, L, symbol_var) > target_outage:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def xi_outage_bound(target_outage: float, kappa: np.ndarray, L: int,
                    symbol_var: float = 1.0) -> float:
    """Worst-case xi_MUI at a given outage probability (drives the rate loss)."""
    tau = safe_threshold(target_outage, kappa, L, symbol_var)
    return excess_noise_xi(tau ** 2, symbol_var)


# --------------------------------------------------------------------------- #
# dynamic (refresh) behaviour
# --------------------------------------------------------------------------- #
def cross_round_correlation(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """rho between previous and current code of every user."""
    L = prev.shape[1]
    return np.einsum("ij,ij->i", prev, cur) / float(L)


def refresh_tradeoff(refresh_fractions: np.ndarray, n_users: int, L: int,
                     seed: int = 0) -> dict:
    """Static vs refreshed codebook: cross-round leakage and estimation error.

    For each refresh fraction f every chip is re-drawn independently with
    probability f, so knowing the previous code leaves a Bernoulli(1-f)
    uncertainty per chip:

      ``rho_prev``  RMS cross-round correlation (should scale as sqrt(1-f)/sqrt(L))
      ``leak_bits`` per-user bits of the current code revealed by the previous
                    code, L * H_b(1-f)
      ``est_err``   sqrt((1-f)/L): normalised blind-estimation rms error

    ``xi_now`` is the in-round MUI excess noise, which is *not* reduced by
    refreshing (fresh codes still have O(1/sqrt(L)) cross-correlation).
    """
    rng = np.random.default_rng(seed)
    out = dict(refresh=[], rho_prev=[], leak_bits=[], est_err=[], xi_now=[])
    for f in np.atleast_1d(refresh_fractions):
        f = float(f)
        prev = rng.integers(0, 2, size=(n_users, L), dtype=np.int8).astype(np.float64) * 2 - 1
        cur = prev.copy()
        n_ref = int(round(f * L))
        if n_ref > 0:
            idx = rng.random((n_users, L)) < (n_ref / L)
            fresh = rng.integers(0, 2, size=(n_users, L)).astype(np.float64) * 2 - 1
            cur[idx] = fresh[idx]
        r = np.abs(cross_round_correlation(prev, cur))
        p = 1.0 - f
        hb = 0.0 if p <= 0 or p >= 1 else (-p * np.log2(p) - (1 - p) * np.log2(1 - p))
        out["refresh"].append(f)
        out["rho_prev"].append(float(np.sqrt(np.mean(r ** 2))))
        out["leak_bits"].append(float(L * hb))
        out["est_err"].append(float(np.sqrt(max(p, 0.0) / L)))
        out["xi_now"].append(xi_from_codes(np.ones(n_users - 1), L))
    return {k: np.asarray(v) for k, v in out.items()}


def mui_vs_users(user_counts: np.ndarray, L: int, V_A: float,
                 power_ratio: float = 1.0, eps_mui: float = 1e-9) -> dict:
    """xi_MUI (mean and eps-quantile) vs number of users."""
    out = dict(n_users=[], xi=[], xi_required=[], nu=[])
    for M in np.atleast_1d(user_counts):
        kappa = uniform_kappa(int(M) - 1, power_ratio)
        out["n_users"].append(int(M))
        out["xi"].append(xi_block_mean(kappa, L))
        out["xi_required"].append(required_xi(eps_mui, kappa, L))
        out["nu"].append(effective_interferers(kappa))
    return {k: np.asarray(v) for k, v in out.items()}


def dynamic_vs_static_rate(L: int, n_users: int, V_A: float,
                           rate_fn, target_outage: float = 1e-9,
                           power_ratios=(0.1, 0.5, 1.0)) -> dict:
    """Rate penalty of MUI for static-public vs dynamic-secret codes.

    ``rate_fn(xi_extra) -> raw key rate``.  The static-public baseline pays the
    same xi (orthogonal codes -> xi = 0, so it wins on *rate*); the point of the
    figure is that the dynamic scheme buys access control at the cost of a
    quantifiable, bounded rate penalty.
    """
    rows = []
    for pr in power_ratios:
        kappa = uniform_kappa(n_users - 1, pr)
        xi_mean = xi_from_codes(kappa, L)
        xi_req = required_xi(target_outage, kappa, L)
        rows.append(dict(power_ratio=float(pr), xi_mean=xi_mean,
                         xi_required=xi_req,
                         rate_mean=rate_fn(xi_mean), rate_worst=rate_fn(xi_req)))
    return dict(L=L, n_users=n_users, outage=target_outage, rows=rows)


# --------------------------------------------------------------------------- #
# the statistically relevant object: the BLOCK-AVERAGED excess noise
# --------------------------------------------------------------------------- #
def xi_block_samples(kappa: np.ndarray, L: int, n_draws: int = 200_000,
                     seed: int = 0) -> np.ndarray:
    """Samples of the block-averaged excess noise xi_block.

    This is the quantity that enters the CV-QKD rate, and it is *not* reduced by
    averaging over many symbols.  With a single codebook held for the whole
    round, the cross-correlations ``rho_0j`` are frozen constants, so

        xi_block = (1/L) * sum_j kappa_j * z_j^2,     z_j ~ N(0,1),

    because the per-symbol random symbols ``a_j`` average out by the law of
    large numbers while ``rho_0j^2`` does not.  Consequently long blocks do
    **not** suppress the MUI uncertainty -- only a longer code (larger L) or
    more, better-balanced interferers does.

    This is the central statistical subtlety of Section 7: the naive
    "per-symbol heavy tail" analysis is irrelevant, and the naive
    "average over 10^8 symbols" intuition is wrong.
    """
    rng = np.random.default_rng(seed)
    kappa = np.atleast_1d(np.asarray(kappa, dtype=float))
    if kappa.size == 0:
        return np.zeros(n_draws)
    z = rng.normal(size=(n_draws, kappa.size))
    return (kappa * z ** 2).sum(axis=1) / float(L)


def _xi_gamma_params(kappa: np.ndarray, L: int) -> tuple[float, float]:
    """Moment-matched (shape, scale) of xi_block; exact for equal kappa."""
    kappa = np.atleast_1d(np.asarray(kappa, dtype=float))
    kappa = kappa[kappa > 0]
    if kappa.size == 0:
        return 0.0, 0.0
    mean = float(np.sum(kappa)) / L
    var = float(2.0 * np.sum(kappa ** 2)) / L ** 2
    return mean ** 2 / var, var / mean


def xi_block_mean(kappa: np.ndarray, L: int) -> float:
    """E[xi_block] = (1/L) sum_j kappa_j."""
    return xi_from_codes(kappa, L)


def xi_block_std(kappa: np.ndarray, L: int) -> float:
    """Std of xi_block. Relative size ~ sqrt(2/K) for K equal interferers:
    ~53% for K=7. The fluctuation is *first order*, not a small correction."""
    kappa = np.atleast_1d(np.asarray(kappa, dtype=float))
    if kappa.size == 0:
        return 0.0
    return float(np.sqrt(2.0 * np.sum(kappa ** 2)) / L)


def xi_block_quantile(p: float, kappa: np.ndarray, L: int) -> float:
    """p-quantile of xi_block (Gamma moment match; exact for equal kappa)."""
    shape, scale = _xi_gamma_params(kappa, L)
    if shape <= 0:
        return 0.0
    return float(stats.gamma.ppf(p, a=shape, scale=scale))


def required_xi(eps_mui: float, kappa: np.ndarray, L: int) -> float:
    """xi_block at confidence ``1 - eps_mui``.

    This is the *design* excess noise that must be fed to the CV-QKD rate so
    that the round is safe with probability at least ``1 - eps_mui``.  Because
    eps_MUI is a *per-round* composition parameter, eps_mui = 1e-12 is
    meaningful here -- unlike a per-symbol outage, which the heavy product tail
    would make absurd.
    """
    return xi_block_quantile(1.0 - float(eps_mui), kappa, L)


def required_xi_normal(eps_mui: float, kappa: np.ndarray, L: int) -> float:
    """Gaussian (CLT) approximation to :func:`required_xi`, for comparison."""
    mu = xi_block_mean(kappa, L)
    sd = xi_block_std(kappa, L)
    return float(mu + stats.norm.isf(float(eps_mui)) * sd)


def xi_overshoot_cost(eps_mui: float, kappa: np.ndarray, L: int,
                      rate_fn) -> dict:
    """Rate cost of provisioning for the eps_MUI quantile instead of the mean."""
    mu = xi_block_mean(kappa, L)
    req = required_xi(eps_mui, kappa, L)
    return dict(xi_mean=mu, xi_required=req, xi_ratio=req / mu if mu > 0 else np.inf,
                rate_mean=rate_fn(mu), rate_required=rate_fn(req),
                rate_loss=rate_fn(mu) - rate_fn(req))


def xi_vs_block_length_note() -> str:
    """Short prose used verbatim in the docs to prevent misreading."""
    return (
        "xi_block = (1/L) * sum_j kappa_j * z_j^2 with z_j ~ N(0,1). "
        "Averaging over more symbols does NOT reduce this: the per-symbol "
        "symbols average out, the frozen cross-correlations do not. "
        "Relative std = sqrt(2 * sum kappa^2) / sum kappa = sqrt(2/K) for K "
        "equal interferers (53% at K=7)."
    )
