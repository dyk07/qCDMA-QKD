"""GG02 coherent-state CV-QKD: asymptotic and finite-size secret-key rate.

Model
-----
Gaussian-modulated coherent states, reverse reconciliation, trusted detector.
All variances in shot-noise units (SNU).

Notation
--------
V_A   : Alice's modulation variance (SNU)
V     : total Alice variance, V = V_A + 1
T     : total transmittance, T = eta * 10**(-alpha*L/10)
xi    : excess noise referred to the channel input (SNU)
chi_line = (1 - T)/T + xi
chi_hom  = (1 + v_el)/eta - 1
chi_het  = 2*(1 + v_el)/eta - 1
chi_tot  = chi_line + chi_det/T

Secret-key rate (asymptotic, reverse reconciliation):
    S = beta * I_AB - chi_BE          [bits / symbol / quadrature]

References (formulas, not code):
  Lodewyck et al., PRA 76, 042305 (2007)
  Fossier et al., J. Phys. B 42, 114014 (2009)
  Laudenbach et al., Adv. Quantum Technol. 1, 1800051 (2018)
  Leverrier et al., PRA 81, 062343 (2010)   [finite-size]
  Ruppert et al., Quantum 6, 622 (2022)     [finite-size, modern notation]

The finite-size correction implemented here is the *standard AEP + PE* form.
It is deliberately simple: the paper's contribution is the closed loop, not a
new finite-key CV-QKD analysis.
"""

from __future__ import annotations

import numpy as np

from .config import CVQKDConfig

LN2 = np.log(2.0)


# --------------------------------------------------------------------------- #
# elementary functions
# --------------------------------------------------------------------------- #
def db_to_transmittance(distance_km: float, alpha_db_per_km: float = 0.2) -> float:
    """*Fibre* transmittance for a given length (excludes detector efficiency)."""
    return 10.0 ** (-alpha_db_per_km * distance_km / 10.0)


def total_transmittance(distance_km: float, alpha_db_per_km: float,
                        eta: float) -> float:
    """Total transmittance ``T = eta * 10**(-alpha*L/10)``.

    This is the ``T`` that every rate function in this module expects.  Keeping
    the detector efficiency in exactly one place avoids the classic factor-eta
    error between the asymptotic and finite-size code paths.
    """
    return eta * db_to_transmittance(distance_km, alpha_db_per_km)


def g(x: np.ndarray | float) -> np.ndarray | float:
    """von Neumann entropy of a thermal state, G(x) = (x+1)log2(x+1) - x log2(x)."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    m = x > 0
    out[m] = (x[m] + 1.0) * np.log2(x[m] + 1.0) - x[m] * np.log2(x[m])
    return out if out.ndim else float(out)


def detector_noise(theta: float, eta: float, v_el: float) -> float:
    """Detector-added noise chi_det for an equivalent one-quadrature measurement.

    theta = 1 (homodyne) or 2 (heterodyne).
    """
    return theta * (1.0 + v_el) / eta - 1.0


def total_noise(T: float, xi: float, eta: float, v_el: float, theta: float) -> float:
    """chi_tot = chi_line + chi_det/T, i.e. all noise referred to Bob's measurement."""
    chi_line = (1.0 - T) / T + xi
    return chi_line + detector_noise(theta, eta, v_el) / T


# --------------------------------------------------------------------------- #
# rate terms
# --------------------------------------------------------------------------- #
def mutual_information(V_A: float, T: float, xi: float, eta: float, v_el: float,
                       detection: str = "homodyne") -> float:
    """I_AB in bits per symbol (per quadrature for homodyne, joint for heterodyne)."""
    V = V_A + 1.0
    theta = 1.0 if detection == "homodyne" else 2.0
    chi_tot = total_noise(T, xi, eta, v_el, theta)
    i1 = 0.5 * np.log2((V + chi_tot) / (1.0 + chi_tot))
    # heterodyne: Bob measures both quadratures -> factor 2 (the theta in chi_tot
    # already accounts for the doubled detector noise).
    return float(i1 if detection == "homodyne" else 2.0 * i1)


def holevo_bound(V_A: float, T: float, xi: float, eta: float, v_el: float,
                 detection: str = "homodyne") -> float:
    """chi_BE: Holevo information of Eve's state conditioned on Bob's measurement
    (reverse reconciliation), bits per symbol.

    Homodyne (Lodewyck et al. 2007, App. A):

        chi_BE = G((l1-1)/2) + G((l2-1)/2) - G((l3-1)/2) - G((l4-1)/2)

    with l1,2 the symplectic eigenvalues of Eve's conditional covariance matrix
    and l3,4 those of the state shared with Bob.  The **minus** signs on l3,4 are
    essential: they encode the conditioning on Bob's measurement.

    Heterodyne (Fossier et al. 2009) uses five eigenvalues:

        chi_BE = G((l1-1)/2)+G((l2-1)/2)-G((l3-1)/2)-G((l4-1)/2)-G((l5-1)/2)
    """
    V = V_A + 1.0
    chi_line = (1.0 - T) / T + xi

    if detection == "homodyne":
        chi_det = detector_noise(1.0, eta, v_el)
        chi_tot = chi_line + chi_det / T

        A = V ** 2 * (1.0 - 2.0 * T) + 2.0 * T + T ** 2 * (V + chi_line) ** 2
        B = T ** 2 * (V * chi_line + 1.0) ** 2
        lam12 = np.sqrt(0.5 * (A + np.array([1.0, -1.0])
                               * np.sqrt(np.maximum(A ** 2 - 4.0 * B, 0.0))))

        C = (A * chi_det + V * np.sqrt(B) + T * (V + chi_line)) / (T * (V + chi_tot))
        D = np.sqrt(B) * (V + np.sqrt(B) * chi_det) / (T * (V + chi_tot))
        lam34 = np.sqrt(0.5 * (C + np.array([1.0, -1.0])
                               * np.sqrt(np.maximum(C ** 2 - 4.0 * D, 0.0))))
        # symplectic eigenvalues: l3,4 are <= ... clip at 1 for numerical safety
        l3 = np.maximum(lam34[0], 1.0)
        l4 = np.maximum(lam34[1], 1.0)
        l1 = np.maximum(lam12[0], 1.0)
        l2 = np.maximum(lam12[1], 1.0)
        return float(g((l1 - 1) / 2) + g((l2 - 1) / 2)
                     - g((l3 - 1) / 2) - g((l4 - 1) / 2))

    # ------------------------------------------------------------------ #
    # heterodyne
    # ------------------------------------------------------------------ #
    raise NotImplementedError(
        "Heterodyne Holevo bound is intentionally not implemented: the "
        "five-symplectic-eigenvalue form was not reproduced against a trusted "
        "reference in this codebase, and shipping a mis-signed expression would "
        "silently overstate the key rate. All experiments use homodyne "
        "detection, which is validated against Lodewyck et al. (2007). "
        "See docs/03_cvqkd_model.md."
    )


def asymptotic_rate(V_A: float, T: float, xi: float, eta: float, v_el: float,
                    beta: float, detection: str = "homodyne") -> dict:
    """Return the ingredients of the asymptotic secret-key rate."""
    i_ab = mutual_information(V_A, T, xi, eta, v_el, detection)
    chi_be = holevo_bound(V_A, T, xi, eta, v_el, detection)
    s = beta * i_ab - chi_be
    return dict(I_AB=i_ab, chi_BE=chi_be, S=s, S_pos=max(s, 0.0))


# --------------------------------------------------------------------------- #
# finite-size
# --------------------------------------------------------------------------- #
def aep_penalty(n: int, eps_sm: float) -> float:
    """AEP / smooth-min-entropy penalty (bits per symbol).

    Delta = 7*sqrt(2)*sqrt(log2(2/eps_sm)/n), cf. Leverrier et al. (2010).
    """
    if n <= 0:
        return np.inf
    return 7.0 * np.sqrt(2.0) * np.sqrt(np.log2(2.0 / eps_sm) / n)


def pe_penalty(n: int, eps_sm: float, eps_pa: float) -> float:
    """Privacy-amplification overhead (bits per block)."""
    return 2.0 * np.log2(1.0 / eps_pa) + np.log2(2.0 / eps_sm)


def finite_size_rate(V_A: float, T: float, xi: float, eta: float, v_el: float,
                     beta: float, detection: str = "homodyne",
                     n_symbols: int = 10 ** 8, eps_sm: float = 1e-10,
                     eps_pa: float = 1e-12, sample_fraction: float = 0.1) -> dict:
    """Finite-size secret-key rate in bits per *total* symbol.

    n_symbols  : total exchanged symbols (including PE samples)
    sample_fraction : fraction of symbols used for parameter estimation
    """
    n_pe = int(sample_fraction * n_symbols)
    n = n_symbols - n_pe                      # symbols usable for the key
    if n <= 0:
        return dict(S=0.0, S_pos=0.0, n_key_symbols=0, I_AB=0.0, chi_BE=0.0,
                    Delta=0.0, overhead=0.0)
    base = asymptotic_rate(V_A, T, xi, eta, v_el, beta, detection)
    delta = aep_penalty(n, eps_sm)
    core = beta * base["I_AB"] - base["chi_BE"] - delta
    overhead = pe_penalty(n, eps_sm, eps_pa) / n_symbols
    s = (n / n_symbols) * core - overhead
    return dict(S=s, S_pos=max(s, 0.0), n_key_symbols=n, I_AB=base["I_AB"],
                chi_BE=base["chi_BE"], Delta=delta, overhead=overhead)


# --------------------------------------------------------------------------- #
# parameter estimation
# --------------------------------------------------------------------------- #
def pe_confidence_intervals(T: float, xi: float, n_pe: int, eps_pe: float = 1e-10,
                            sample_fraction: float = 0.1) -> dict:
    """Conservative PE bounds.

    Uses a normal approximation on the empirical variance of the PE sample:
    the relative standard error of a variance estimate from n_pe samples is
    sqrt(2/n_pe). We inflate both noise and transmittance by
    z = sqrt(2 ln(1/eps_pe)):
        xi_ub   = xi + z * sqrt(2/n_pe) * (1 + xi)
        T_lb    = T  * (1 - z * sqrt(2/n_pe))
    """
    if n_pe <= 0:
        return dict(T_lb=0.0, xi_ub=np.inf, z=np.inf, rel_err=np.inf)
    z = np.sqrt(2.0 * np.log(1.0 / eps_pe))
    rel = z * np.sqrt(2.0 / n_pe)
    return dict(T_lb=max(T * (1.0 - rel), 0.0),
                xi_ub=xi + rel * (1.0 + xi),
                z=float(z), rel_err=float(rel))


# --------------------------------------------------------------------------- #
# convenience: distance sweep
# --------------------------------------------------------------------------- #
def rate_vs_distance(cfg: CVQKDConfig, distances_km: np.ndarray,
                     extra_noise: float = 0.0) -> np.ndarray:
    """Asymptotic rate (bits/symbol) for each distance.

    ``extra_noise`` is added to the excess noise (e.g. MUI, code leakage).
    """
    out = np.empty(np.atleast_1d(distances_km).shape, dtype=float)
    for i, L in enumerate(np.atleast_1d(distances_km)):
        T = total_transmittance(float(L), cfg.alpha_db_per_km, cfg.eta)
        out.flat[i] = asymptotic_rate(cfg.V_A, T, cfg.excess_noise + extra_noise,
                                      cfg.eta, cfg.v_el, cfg.beta,
                                      cfg.detection)["S_pos"]
    return out


def max_distance(cfg: CVQKDConfig, extra_noise: float = 0.0,
                 hi_km: float = 300.0, tol: float = 1e-6) -> float:
    """Distance at which the asymptotic rate vanishes (bisection)."""
    lo = 0.0
    T_hi = total_transmittance(hi_km, cfg.alpha_db_per_km, cfg.eta)
    if asymptotic_rate(cfg.V_A, T_hi, cfg.excess_noise + extra_noise, cfg.eta,
                       cfg.v_el, cfg.beta, cfg.detection)["S_pos"] > tol:
        return hi_km
    for _ in range(100):
        mid = 0.5 * (lo + hi_km)
        T = total_transmittance(mid, cfg.alpha_db_per_km, cfg.eta)
        r = asymptotic_rate(cfg.V_A, T, cfg.excess_noise + extra_noise,
                            cfg.eta, cfg.v_el, cfg.beta, cfg.detection)["S_pos"]
        if r > tol:
            lo = mid
        else:
            hi_km = mid
    return 0.5 * (lo + hi_km)
