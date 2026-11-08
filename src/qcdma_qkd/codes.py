"""q-CDMA spreading codes: generation and cross-correlation statistics.

Two families are modelled:

``random``
    i.i.d. +-1 chips (``refresh_fraction`` of the chips re-drawn every round).
    Normalised cross-correlation between two independent random codes of length
    L behaves as  rho ~ N(0, 1/L)  (exactly: (1/L) * sum of L i.i.d. signs).

``walsh``
    rows of a Hadamard matrix (exactly orthogonal). This is the "ideal static"
    baseline: zero cross-correlation by construction, but the codebook is fixed
    and public, hence offers no access control.

The interplay between code refresh and residual cross-correlation is the object
of the "dynamic MUI" analysis (Section 7 of the paper outline).
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
def random_code(length: int, rng: np.random.Generator) -> np.ndarray:
    """i.i.d. +-1 code of the given length."""
    return rng.integers(0, 2, size=length, dtype=np.int8) * 2 - 1


def random_codebook(n_users: int, length: int, rng: np.random.Generator) -> np.ndarray:
    """(n_users, length) matrix of independent random +-1 codes."""
    return random_code((n_users, length), rng).astype(np.float64)


def hadamard_codebook(n_users: int, length: int) -> np.ndarray:
    """First ``n_users`` rows of a normalised Walsh-Hadamard codebook."""
    if length & (length - 1):
        raise ValueError("length must be a power of two for Walsh codes")
    from scipy.linalg import hadamard
    H = hadamard(length, dtype=np.float64)
    if n_users > length:
        raise ValueError("cannot have more users than chips in a Hadamard codebook")
    return H[:n_users]


def refreshed_codebook(base: np.ndarray, refresh_fraction: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Re-randomise a fraction of the chips of every code.

    ``refresh_fraction`` 0 -> static codebook, 1 -> fully fresh codebook.
    """
    if not 0.0 <= refresh_fraction <= 1.0:
        raise ValueError("refresh_fraction must be in [0, 1]")
    n_users, L = base.shape
    n_ref = int(round(refresh_fraction * L))
    out = base.copy()
    if n_ref == 0:
        return out
    idx = rng.random((n_users, L)) < (n_ref / L)
    fresh = random_code((n_users, L), rng).astype(np.float64)
    out[idx] = fresh[idx]
    return out


# --------------------------------------------------------------------------- #
# cross-correlation
# --------------------------------------------------------------------------- #
def normalised_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """rho = <a,b>/L for +-1 sequences (O(1/L) resolution)."""
    return float(np.dot(a, b) / a.size)


def correlation_matrix(codebook: np.ndarray) -> np.ndarray:
    """R_ij = <c_i, c_j>/L, unit diagonal."""
    L = codebook.shape[1]
    return (codebook @ codebook.T) / L


def offdiag_abs_max(codebook: np.ndarray) -> float:
    R = np.abs(correlation_matrix(codebook))
    np.fill_diagonal(R, 0.0)
    return float(R.max())


def observed_correlation_samples(codebook: np.ndarray, n_pairs: int | None = None,
                                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Return the multiset of off-diagonal correlations (all pairs by default)."""
    R = correlation_matrix(codebook)
    n = R.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = R[iu]
    if n_pairs is not None and n_pairs < vals.size:
        rng = rng or np.random.default_rng(0)
        vals = rng.choice(vals, size=n_pairs, replace=False)
    return vals


# --------------------------------------------------------------------------- #
# blind code estimation (attacker side)
# --------------------------------------------------------------------------- #
def guess_entropy_given_observations(seed_bits: int, n_observations: float,
                                     chips_per_observation: float) -> float:
    """Remaining entropy (bits) of a code seed after Eve observed n_observations.

    Simplified model: every observation of ``chips_per_observation`` chips leaks
    at most ``chips_per_observation`` bits (a +-1 chip carries 1 bit when the
    symbol is unknown-but-Gaussian, less when the symbol is known). We cap the
    leakage at the seed size:

        H_remaining = max(0, seed_bits - n_observations * chips_per_observation)

    Returns 0 once the seed is fully determined. This is the conservative
    "Eve learns chips linearly" model used in Section 5; the true information
    theoretic blind-despreading bound is the open problem flagged in the outline.
    """
    leaked = min(float(seed_bits), n_observations * chips_per_observation)
    return float(max(0.0, seed_bits - leaked))


def successful_estimation_probability(seed_bits: int, leak_fraction: float) -> float:
    """Probability that Eve reconstructs a seed from a fraction ``leak_fraction``.

    Uses a guessing-exponent ansatz  P_success = 2^{-(1-leak_fraction)*seed_bits}.
    """
    leak_fraction = float(np.clip(leak_fraction, 0.0, 1.0))
    return float(2.0 ** (-(1.0 - leak_fraction) * seed_bits))
