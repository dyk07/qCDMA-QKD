"""E11: the covariance (subspace) attack -- code secrecy under a collective Eve.

What this experiment establishes
--------------------------------
The blind-code-estimation channel was modelled per chip (E4/E8/05 §3.2).  That
premise is wrong.  With ``+-1`` codes and Gaussian symbols the received vector
satisfies ``y_t | C ~ N(0, Sigma)`` **exactly**, with

    Sigma = sigma^2 sum_j kappa_j c_j c_j^T + n_v I ,

so the observation law depends on the codebook *only through Sigma*.  The signs
of the interferers' symbols never enter it.  Eve therefore does not have to
resolve anything chip by chip: ``S_hat = (1/n) sum y y^T`` is a sufficient
statistic, and the question becomes "how many symbols does the subspace need?".

Panels
------
1. Sample cost of the two attacks vs the weak interferer's power.  Honest
   result: at equal physical power the **matched filter is cheaper in count**.
   The subspace attack is not the cheaper attack -- it is the one with the
   weaker *premise*, which is the point that matters here.
2. Measured recovery fraction vs ``n`` for equal / spread / weak powers --
   the simulated attack end to end (:func:`attack.code_recovery_from_covariance`).
3. Residual ambiguity after a perfect Sigma: ``log2(M!)`` bits, independent of
   the seed, vs the chip-based model's ``seed - k``.

Outputs
-------
results/figures/e11_covariance_attack.png
results/data/e11_covariance_attack.json
"""

from __future__ import annotations

import numpy as np

from _common import dump, header, scenario

from qcdma_qkd import attack, mui, plotting

L = 1024
M = 8
N_GRID = (10 ** 4, 10 ** 5, 10 ** 6)
KAPPAS = {
    "equal": np.ones(M),
    "spread": np.linspace(1.0, 0.3, M),
    "weak": np.concatenate([np.ones(M - 1), [0.01]]),
}


def main() -> dict:
    header("E11  Covariance attack: code secrecy from second moments alone")
    sc = scenario()
    cdma = sc["cdma"]
    seed_bits = int(sc["budget"].seed_bits)
    Lx = int(cdma.spreading_factor) if L is None else L

    # ---------------------------------------------------------------- #
    # 1. Cost of the two blind attacks vs the weak interferer's power
    # ---------------------------------------------------------------- #
    powers = np.logspace(-4, 1, 25)
    cost = []
    for p in powers:
        kappa = np.concatenate([np.ones(M - 1), [p]])
        env = attack.blind_attack_envelope(Lx, kappa, per_chip_snr=p,
                                           target_p_chip=0.99)
        cost.append(dict(power=float(p), **{k: (float(v) if not isinstance(v, bool)
                                                else v)
                                            for k, v in env.items()}))
    crossing = next((c["power"] for c in cost if not c["chip_is_cheaper"]), None)

    # ---------------------------------------------------------------- #
    # 2. Measured end-to-end recovery
    # ---------------------------------------------------------------- #
    rng = np.random.default_rng(20261109)
    C = rng.choice([-1.0, 1.0], size=(M, Lx)).astype(float)

    recovery = {}
    for name, kappa in KAPPAS.items():
        rows = []
        for n in N_GRID:
            r = attack.code_recovery_from_covariance(
                C, kappa, n_samples=int(n), chunk=25000,
                restarts=200, iters=2000, tol=0.05, seed=3)
            rows.append(r)
            print(f"  [{name}] n={n:.0e}  err={r['err_operator_norm']:7.2f}  "
                  f"found={r['n_found']:3d}  rec={r['recovery_fraction']:.3f}")
        recovery[name] = rows

    # ---------------------------------------------------------------- #
    # 3. Residual ambiguity: log2(M!) vs the chip model's seed - k
    # ---------------------------------------------------------------- #
    ambiguity = attack.code_ambiguity_bits(M, equal_powers=True)
    spec = attack.observation_covariance(np.ones(M), Lx)
    k_req = attack.required_entropy_bits(1e-12)
    chip_model_h = attack.max_tolerable_known_chips(seed_bits, 1e-12)
    eps_from_ambiguity = 2.0 ** (-ambiguity)

    total_symbols = 10 ** 8
    env_by_kappa = {k: attack.blind_attack_envelope(Lx, v, per_chip_snr=0.5)
                    for k, v in KAPPAS.items()}
    out = dict(
        L=Lx, M=M, seed_bits=seed_bits, n_grid=list(N_GRID),
        kappas={k: [float(x) for x in v] for k, v in KAPPAS.items()},
        cost=cost, crossing_power=crossing,
        recovery=recovery,
        ambiguity_bits=float(ambiguity),
        eps_code_from_ambiguity=float(eps_from_ambiguity),
        k_req=k_req, chip_model_max_known=float(chip_model_h),
        snr_eff_equal=spec["snr_eff"],
        n_theory={k: float(attack.subspace_identification_samples(Lx, v))
                  for k, v in KAPPAS.items()},
        total_symbols_per_round=int(total_symbols),
        fraction_of_round={k: float(v["n_samples"] / total_symbols)
                           for k, v in env_by_kappa.items()},
    )

    # ---------------------------------------------------------------- #
    # figure
    # ---------------------------------------------------------------- #
    fig, axes = plotting.plt.subplots(1, 3, figsize=(12.6, 3.4))

    ax = axes[0]
    ax.loglog(powers, [c["n_chip"] for c in cost], "o-", ms=3,
              color="#c0392b", label="per-chip (matched filter)")
    ax.loglog(powers, [c["n_subspace"] for c in cost], "s-", ms=3,
              color="#2471a3", label="subspace (second moments)")
    ax.set_xlabel("weak interferer power  $\\kappa$")
    ax.set_ylabel("symbols needed")
    ax.set_title("Both attacks, same physical power")
    ax.legend(fontsize=7)
    if crossing is not None:
        ax.axvline(crossing, color="k", ls=":", lw=0.8)

    ax = axes[1]
    for name, col in zip(KAPPAS, ("#c0392b", "#2471a3", "#1e8449")):
        ys = [r["recovery_fraction"] for r in recovery[name]]
        ax.semilogx(N_GRID, ys, "o-", color=col, label=name)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("symbols observed  $n$")
    ax.set_ylabel("codes recovered")
    ax.set_title("Measured recovery from $\\hat S$")
    ax.legend(fontsize=7)

    ax = axes[2]
    labels = ["chip model\n$seed-k_{req}$", "measured\n$\\log_2 M!$"]
    vals = [chip_model_h, ambiguity]
    ax.bar(labels, vals, color=("#d68910", "#1e8449"), alpha=0.85)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.1f}", (i, v), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("bits Eve still lacks")
    ax.set_title("Residual ambiguity after a perfect $\\Sigma$")

    plotting.save(fig, "e11_covariance_attack")
    return out


if __name__ == "__main__":
    dump("e11_covariance_attack", main())
