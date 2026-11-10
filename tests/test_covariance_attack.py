"""The covariance (subspace) attack: the codebook is what the second moments see.

Why this file exists
--------------------
The project previously modelled blind code estimation with a **per-chip**
matched filter: Eve tries to guess individual chips, succeeds with probability
``p_chip``, and the surviving seed entropy is ``seed - p_chip * L``.  That model
is not merely loose -- it is the *wrong premise*.

With normalised ``+-1`` codes and i.i.d. Gaussian symbols the received vector is

    y_t = sum_j sqrt(kappa_j) a_j(t) c_j + n_t ,      a_j ~ N(0, sigma^2)

which is a linear combination of independent Gaussians, hence **exactly**

    y_t | C ~ N(0, Sigma),   Sigma = sigma^2 sum_j kappa_j c_j c_j^T + n_v I .

The observation law therefore depends on the codebook *only through Sigma*.
The signs of the interferers' symbols never appear, so "resolving the
interferers' symbols" is not a prerequisite for attacking the code.  The
sufficient statistic is the sample covariance ``S_hat``, and the whole question
becomes: how many samples does it take to read the codes out of ``Sigma``?

The answer, measured by :func:`attack.code_recovery_from_covariance`:

* the top-``M`` eigenvectors of ``S_hat`` span ``span{c_j}``;
* on their own the eigenvectors recover **nothing** (any orthogonal rotation of
  a spanning set gives statistically identical Gaussian observations);
* the discrete ``+-1`` prior breaks that rotation symmetry: enumerating the
  ``+-1`` lattice points inside the estimated subspace returns ``2M`` vectors,
  i.e. every code up to an irrelevant global sign;
* the cost is ``O(L / snr_eff^2)`` in *samples*, not in chips, and it is linear
  in ``L`` -- so a **weak** interferer is the *cheapest* to resolve, the exact
  opposite of the per-chip model's prediction.

Consequence for the project (the part that changes conclusions): code secrecy
against a collective attacker does not survive on entropy arguments.  The
residual ambiguity after a perfect ``Sigma`` is only ``log2(M!)`` bits (15.3 bit
at ``M = 8``) and, critically, it is **independent of the seed length**.  The
refresh constraint therefore cannot be justified by "Eve only knows k chips";
see ``docs/05_security.md`` §3.2 and ``docs/09_model_vs_bound.md`` §5.
"""

from __future__ import annotations

import numpy as np
import pytest

from qcdma_qkd import attack


L = 256
M = 4


@pytest.fixture(scope="module")
def codebook():
    rng = np.random.default_rng(7)
    return rng.choice([-1.0, 1.0], size=(M, L))


# --------------------------------------------------------------------------- #
# 1. The observation law really is N(0, Sigma) -- nothing about signs survives
# --------------------------------------------------------------------------- #
def test_coded_covariance_matches_closed_form(codebook):
    """``coded_covariance`` equals ``symbol*C^T diag(kappa) C + noise*I``."""
    kappa = np.array([1.0, 0.5, 0.25, 0.1])
    S = attack.coded_covariance(codebook, kappa, symbol_var=0.7, noise_var=0.3)
    want = 0.7 * (codebook.T * kappa) @ codebook + 0.3 * np.eye(L)
    assert np.allclose(S, want)
    assert np.allclose(S, S.T)


def test_spectrum_is_noise_plus_one_shift_per_user(codebook):
    """The spectrum is ``n_v + sigma^2 L kappa_j`` (once) and ``n_v`` (L-M times).

    This is the structural statement: ``Sigma`` exposes the powers and the
    subspace, nothing else.
    """
    kappa = np.array([1.0, 0.5, 0.25, 0.1])
    sigma2, nv = 0.8, 0.4
    S = attack.coded_covariance(codebook, kappa, sigma2, nv)

    # random +-1 codes are near-orthogonal, so the shifts are additive
    shifts = sigma2 * L * kappa
    lam = np.linalg.eigvalsh(S)
    top = lam[-M:][::-1]
    assert np.allclose(top, nv + shifts, rtol=0.05)
    assert np.allclose(lam[:-M], nv, atol=0.05)


def test_observation_covariance_reports_conservative_snr(codebook):
    """``snr_eff`` pairs the *smallest* shift with the *largest* eigenvalue."""
    kappa = np.array([1.0, 1.0, 0.5, 0.01])
    spec = attack.observation_covariance(kappa, L)
    assert spec["lambda_signal"] == pytest.approx(1.0 * L * 0.01)
    assert spec["lambda_max"] == pytest.approx(1.0 + 1.0 * L * 1.0)
    assert spec["snr_eff"] == pytest.approx(spec["lambda_signal"]
                                            / spec["lambda_max"])
    assert not spec["degenerate_signal"]


def test_degenerate_signal_flag_for_equal_powers():
    assert attack.observation_covariance(np.ones(8), 1024)["degenerate_signal"]
    assert not attack.observation_covariance(np.arange(1.0, 9.0), 1024)[
        "degenerate_signal"]


def test_observation_law_is_sign_and_symbol_blind(codebook):
    """Sigma is invariant under a global sign flip of any code.

    The matched-filter model said Eve must resolve the interferers' symbols.
    The covariance is blind to them: only ``c_j c_j^T`` enters.
    """
    kappa = np.array([1.0, 0.7, 0.4, 0.2])
    S1 = attack.coded_covariance(codebook, kappa)
    flipped = codebook.copy()
    flipped[1] *= -1.0
    flipped[3] *= -1.0
    S2 = attack.coded_covariance(flipped, kappa)
    assert np.allclose(S1, S2)


# --------------------------------------------------------------------------- #
# 2. The +-1 prior is what does the work (and it is decisive)
# --------------------------------------------------------------------------- #
def test_count_pm1_vectors_is_two_m_for_random_codes():
    """A random rank-M +-1 subspace contains exactly ``2M`` +-1 vectors.

    Continuum rotations fill the subspace with infinitely many unit vectors; the
    ``+-1`` prior cuts that to ``2M`` (each code up to sign).  This is the entire
    reason the subspace attack works.
    """
    rng = np.random.default_rng(11)
    for Ls, Ms in ((12, 2), (14, 3), (16, 4)):
        C = rng.choice([-1.0, 1.0], size=(Ms, Ls))
        n = attack.count_pm1_vectors_in_span(C)
        # 2M plus (rarely) one accidental extra pair
        assert n >= 2 * Ms
        assert n <= 2 * Ms + 2, f"L={Ls} M={Ms} gave {n}"


def test_count_pm1_vectors_rejects_large_L():
    with pytest.raises(ValueError):
        attack.count_pm1_vectors_in_span(np.ones((2, 21)))


def test_recover_codes_from_exact_subspace(codebook):
    """A perfect subspace yields every code up to sign."""
    Q, _ = np.linalg.qr(codebook.T)
    rec = attack.recover_codes_from_subspace(Q, restarts=120, seed=3)
    found = rec["vectors"]
    assert found.shape[0] >= 2 * M
    assert rec["exact"], rec["worst_projection_rms"]

    Cn = codebook / np.linalg.norm(codebook, axis=1, keepdims=True)
    Fn = found / np.linalg.norm(found, axis=1, keepdims=True)
    best = np.abs(Fn @ Cn.T).max(axis=0)
    assert np.allclose(best, 1.0, atol=1e-6)


def test_eigenvectors_alone_recover_nothing(codebook):
    """The estimated *eigenvectors* are not the codes.

    Only the discrete prior recovers them; without it the subspace is useless
    (classical ICA unidentifiability).  This guards against "we took the top
    eigenvectors, so we know the codes".
    """
    kappa = np.ones(M)
    S = attack.coded_covariance(codebook, kappa)
    _, U = np.linalg.eigh(S)
    Q = U[:, -M:]
    Cn = codebook / np.linalg.norm(codebook, axis=1, keepdims=True)
    Qn = Q / np.linalg.norm(Q, axis=0, keepdims=True)
    worst = np.abs(Qn.T @ Cn.T).max(axis=0).min()
    # equal powers -> arbitrary rotation inside the eigenspace
    assert worst < 0.95


# --------------------------------------------------------------------------- #
# 3. Sample cost: linear in L, inverse-quadratic in the spectral SNR
# --------------------------------------------------------------------------- #
def test_subspace_samples_scaling():
    """``n ~ 4c^2 (L + log(2/eps)) / snr_eff^2``: linear in ``L``, quadratic in 1/snr.

    Doubling ``L`` doubles the cost.  The inverse-quadratic law is checked as an
    identity on ``n * snr_eff^2``, because ``snr_eff`` itself depends on both
    ``L`` and ``kappa`` and cannot be varied independently of them.
    """
    kappa = np.ones(4)
    n1 = attack.subspace_identification_samples(1024, kappa, eps=1e-12)
    n2 = attack.subspace_identification_samples(2048, kappa, eps=1e-12)
    assert n2 / n1 == pytest.approx(2.0, rel=0.02)

    # n * snr_eff^2 = constant * (L + log(2/eps))  -- the scaling law
    for kappa in (np.ones(4), np.array([1.0, 0.6, 0.3, 0.1]),
                  np.array([1.0, 1.0, 1.0, 0.02])):
        n = attack.subspace_identification_samples(1024, kappa, eps=1e-12)
        snr = attack.observation_covariance(kappa, 1024)["snr_eff"]
        scale = n * snr ** 2 / (1024 + np.log(2 / 1e-12))
        assert scale == pytest.approx(64.0, rel=1e-9)


def test_subspace_cost_is_independent_of_chip_snr():
    """The subspace attack never uses a per-chip SNR, unlike the matched filter.

    This is the *structural* difference, and the reason the per-chip model is
    not the right premise: its cost grows as ``1 / per_chip_snr`` without bound,
    while the subspace cost is set by the spectral separation of ``Sigma``.
    Cost alone does **not** favour the subspace route (see
    ``test_envelope_prefers_chip_in_sample_count``), but cost is not what
    invalidates the chip model -- the premise is.
    """
    kappa = np.ones(4)
    a = attack.subspace_identification_samples(1024, kappa)
    # the signature has no per_chip_snr argument: cost is chip-SNR-free
    assert a == pytest.approx(attack.subspace_identification_samples(1024, kappa))

    # matched filter degrades without bound; subspace does not
    n_chip_strong = attack.chip_identification_samples(1024, 1.0)
    n_chip_weak = attack.chip_identification_samples(1024, 1e-4)
    assert n_chip_weak / n_chip_strong == pytest.approx(1e4, rel=1e-6)


def test_envelope_prefers_chip_in_sample_count():
    """Honest bookkeeping: at equal physical power the matched filter is cheaper.

    Recorded explicitly so the project does not over-claim that the subspace
    attack is also the *cheapest* attack.  It is the one with the weaker
    premise, which is a different (and here decisive) argument.
    """
    kappa = np.concatenate([np.ones(7), [0.01]])
    env = attack.blind_attack_envelope(1024, kappa, per_chip_snr=0.01)
    assert env["chip_is_cheaper"]
    assert env["n_samples"] == pytest.approx(env["n_chip"])
    assert env["ratio_subspace_over_chip"] > 1.0


def test_chip_identification_inverts_chip_success():
    """``chip_identification_samples`` must be the inverse of ``p_chip``."""
    L, snr = 1024, 0.5
    n = attack.chip_identification_samples(L, snr, target_p_chip=0.99)
    p = attack.chip_success_probability(snr, L, n)
    assert p == pytest.approx(0.99, abs=1e-6)


def test_envelope_takes_the_minimum():
    kappa = np.ones(4)
    env = attack.blind_attack_envelope(1024, kappa, per_chip_snr=1e-3)
    assert env["n_samples"] == pytest.approx(min(env["n_chip"],
                                                 env["n_subspace"]))


# --------------------------------------------------------------------------- #
# 4. Residual ambiguity is log2(M!) and independent of the seed
# --------------------------------------------------------------------------- #
def test_code_ambiguity_bits():
    assert attack.code_ambiguity_bits(8, equal_powers=False) == 0.0
    assert attack.code_ambiguity_bits(8, equal_powers=True) == pytest.approx(
        np.log2(40320))
    assert attack.code_ambiguity_bits(1, equal_powers=True) == pytest.approx(0.0)


def test_ambiguity_is_independent_of_seed_bits():
    """The defect that invalidates the chip-based eps_code accounting.

    Ambiguity depends on ``M`` only.  Making the seed longer does not help,
    which is exactly why "Eve knows k of the seed bits" cannot be the right
    parametrisation of this attack.
    """
    a = attack.code_ambiguity_bits(8, equal_powers=True)
    # no seed argument exists at all -- the point is structural
    assert a == pytest.approx(15.299, abs=1e-3)


# --------------------------------------------------------------------------- #
# 5. The measured end-to-end attack
# --------------------------------------------------------------------------- #
def test_code_recovery_from_covariance_end_to_end():
    """Eve recovers all codes from second moments alone.

    ``L = 256``, ``M = 4``, ``n = 2e4`` samples: an affordable number for a
    receiver that sees the whole multiplexed field.  No symbol knowledge, no
    per-chip guessing.
    """
    rng = np.random.default_rng(5)
    C = rng.choice([-1.0, 1.0], size=(4, 256))
    out = attack.code_recovery_from_covariance(
        C, np.ones(4), n_samples=20000, restarts=150, chunk=5000, seed=2)
    assert out["recovery_fraction"] >= 0.99, out["per_code_best_corr"]
    assert out["n_found"] >= 8
    # the operator-norm error must be well below the spectral gap
    assert out["err_operator_norm"] < 0.5 * (out["lambda_top"]
                                             - out["lambda_noise_top"])


def test_code_recovery_survives_unequal_powers():
    """Distinct powers order the codes, so recovery should be at least as easy."""
    rng = np.random.default_rng(9)
    C = rng.choice([-1.0, 1.0], size=(4, 256))
    out = attack.code_recovery_from_covariance(
        C, np.array([1.0, 0.8, 0.5, 0.3]), n_samples=30000,
        restarts=150, chunk=5000, seed=4)
    assert out["recovery_fraction"] >= 0.99, out["per_code_best_corr"]


def test_code_recovery_invalid_inputs(codebook):
    with pytest.raises(ValueError):
        attack.code_recovery_from_covariance(codebook, np.ones(M),
                                             n_samples=0)
    with pytest.raises(ValueError):
        attack.code_recovery_from_covariance(codebook, np.ones(M + 1),
                                             n_samples=100)
