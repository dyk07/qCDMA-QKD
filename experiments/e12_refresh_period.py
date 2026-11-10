"""E12: code refresh period as a three-way trade-off (security / MUI / PRF cost).

Why this experiment exists
--------------------------
E11 showed the codebook is recoverable from second moments in ~1e4 symbols
(``docs/05_security.md`` §3.3-3.4).  The structural reason is that the codebook
is *repeated*: Bob must reuse it to despread, so ``Sigma`` accumulates samples
indefinitely.  If the codebook is re-drawn every ``T`` symbols instead, no single
``Sigma`` gets more than ``T`` samples, and the attack degrades.

But ``T`` is not free:

* **MUI stability** - the project's headline physical result is that the
  block-averaged excess noise has relative fluctuation ``sqrt(2/K)`` = 53%
  that does **not** shrink with block length, precisely because the codes are
  frozen.  Refreshing ``F`` times per block cuts it to ``sqrt(2/(KF))``.
* **PRF throughput** - a refresh costs ``M * L`` bits of PRF output, i.e.
  ``M * L / T`` bits per symbol.

So ``T`` is the single knob trading these three.  This experiment measures all
three on one axis and reports the region where the attack fails.

Important honesty note
----------------------
"Attack fails at short ``T``" is a *statement about the second-moment attack
only*.  It does **not** establish cryptographic security: higher moments, the
PRF structure itself, and correlated-symbol attacks are all unmodelled.  The
numbers here bracket a design point; they do not certify it.

Outputs
-------
results/figures/e12_refresh_period.png
results/data/e12_refresh_period.json
"""

from __future__ import annotations

import numpy as np

from _common import dump, header, scenario

from qcdma_qkd import attack, mui, plotting

L = 1024
M = 8
BLOCK_SYMBOLS = 10 ** 8
SYMBOL_RATE_HZ = 1.0e9 / L          # chip_rate / L
RECOVERY_THRESHOLD = 0.999          # "all codes recovered" means >= this

# Refresh periods to scan (symbols between codebook re-draws).
# 1 = per-symbol, BLOCK_SYMBOLS = frozen (the project baseline).
# The transition is measured to sit between 300 and 500 symbols, so the grid is
# dense there; the wide points on either side show the asymptotes.
T_GRID = (1, 10, 100, 200, 300, 400, 500, 700, 1000,
          3000, 10_000, 100_000, BLOCK_SYMBOLS)

# The threshold is statistical, so a period counts as "blocked" only if *no*
# seed recovers the codes.  Reporting one seed would hide that the attack still
# succeeds on some codebook draws near the transition.
SEEDS = (1, 2, 3, 4)

KAPPAS = {
    "equal": np.ones(M),
    "weak": np.concatenate([np.ones(M - 1), [0.01]]),
}


def main() -> dict:
    header("E12  Code refresh period: security vs MUI stability vs PRF cost")
    sc = scenario()
    cdma = sc["cdma"]
    Lx = int(cdma.spreading_factor) if L is None else L
    Mr = int(cdma.n_users) if M is None else M

    rng = np.random.default_rng(20261110)
    C = rng.choice([-1.0, 1.0], size=(Mr, Lx)).astype(float)

    rows = []
    for T in T_GRID:
        n_frames = max(int(round(BLOCK_SYMBOLS / T)), 1)
        row = dict(refresh_period=int(T), n_frames=n_frames)

        # ---- 1. attack: Eve only gets T samples of one codebook ---------- #
        n_attack = int(min(T, 100_000))       # cap the simulation cost
        rec = {}
        for name, kappa in KAPPAS.items():
            best = None
            for sd in SEEDS:
                r = attack.code_recovery_from_covariance(
                    C, kappa, n_samples=n_attack, chunk=25_000,
                    restarts=200, iters=2000, tol=0.05, seed=sd)
                if best is None or r["recovery_fraction"] > best["recovery_fraction"]:
                    best = r
            rec[name] = best
        row["n_samples_per_codebook"] = n_attack
        row["recovery_fraction"] = {k: v["recovery_fraction"]
                                    for k, v in rec.items()}
        row["n_found"] = {k: v["n_found"] for k, v in rec.items()}
        row["err_operator_norm"] = rec["equal"]["err_operator_norm"]
        row["lambda_top"] = rec["equal"]["lambda_top"]
        # blocked if no configuration reaches full recovery on any seed
        row["attack_blocked"] = bool(
            max(v["recovery_fraction"] for v in rec.values())
            < RECOVERY_THRESHOLD)

        # ---- 2. MUI stability: fluctuation shrinks as 1/sqrt(F) ---------- #
        kappa = mui.uniform_kappa(Mr - 1, 1.0)
        xi = mui.xi_frame_refresh(kappa, Lx, n_frames,
                                  n_symbols_per_frame=int(T), n_codes=Mr)
        row.update(xi_mean=xi["xi_mean"], xi_std=xi["xi_std"],
                   xi_rel_std=xi["xi_rel_std"], xi_eps=xi["xi_eps"])

        # ---- 3. PRF cost -------------------------------------------------- #
        # one refresh = M*L bits; T symbols per refresh at SYMBOL_RATE_HZ
        row["prf_bits_per_refresh"] = float(Mr * Lx)
        row["prf_throughput_bit_per_s"] = (
            float(Mr * Lx) * SYMBOL_RATE_HZ / float(max(int(T), 1)))
        row["refresh_rate_hz"] = SYMBOL_RATE_HZ / float(max(int(T), 1))

        rows.append(row)
        print(f"  T={T:>10d}  n={n_attack:>6d}  "
              f"rec={row['recovery_fraction']['equal']:.3f}  "
              f"blocked={row['attack_blocked']!s:5s}  "
              f"xi_relstd={row['xi_rel_std']:.4f}  "
              f"PRF={row['prf_throughput_bit_per_s'] / 1e6:9.3f} Mbit/s")
    # --- summary: the *largest* T at which the attack is still blocked ---- #
    # T is sorted ascending, so the blocking region is a prefix.  The boundary
    # is the largest blocked T (equivalently: the smallest T that is NOT
    # blocked).  Taking min() here would just return T=1 and say nothing.
    blocked_ts = [r["refresh_period"] for r in rows if r["attack_blocked"]]
    t_blocked_max = max(blocked_ts) if blocked_ts else None
    unblocked_ts = [r["refresh_period"] for r in rows if not r["attack_blocked"]]
    t_unblocked_min = min(unblocked_ts) if unblocked_ts else None
    safe_row = next((r for r in rows if r["refresh_period"] == t_blocked_max),
                    None)
    # Refreshing faster than this is safe against the second-moment attack, at
    # the PRF cost and MUI-fluctuation figure recorded in that row.
    out = dict(
        L=Lx, M=Mr, block_symbols=BLOCK_SYMBOLS,
        symbol_rate_hz=SYMBOL_RATE_HZ,
        recovery_threshold=RECOVERY_THRESHOLD,
        seeds=list(SEEDS),
        kappas={k: [float(x) for x in v] for k, v in KAPPAS.items()},
        rows=rows,
        t_blocked_max=t_blocked_max,
        t_unblocked_min=t_unblocked_min,
        safe_point=safe_row,
        frozen_baseline=rows[-1],
        n_theory_bound=float(attack.subspace_identification_samples(
            Lx, np.ones(Mr))),
        note=("'blocked' refers to the second-moment attack only; no "
              "cryptographic conclusion follows"),
    )

    # ------------------------------------------------------------------ #
    # figure
    # ------------------------------------------------------------------ #
    fig, axes = plotting.plt.subplots(1, 3, figsize=(12.6, 3.4))
    ts = [r["refresh_period"] for r in rows]

    ax = axes[0]
    ax.semilogx(ts, [r["recovery_fraction"]["equal"] for r in rows], "o-",
                color="#c0392b", ms=4, label="equal powers")
    ax.semilogx(ts, [r["recovery_fraction"]["weak"] for r in rows], "s--",
                color="#2471a3", ms=4, label=r"weak $\kappa=0.01$")
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("refresh period $T$ (symbols)")
    ax.set_ylabel("codes recovered")
    ax.set_ylim(-0.05, 1.08)
    ax.set_title("Attack vs refresh period")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.loglog(ts, [r["xi_rel_std"] for r in rows], "o-", color="#1e8449", ms=4)
    ax.axhline(0.535, color="#7f8c8d", ls=":", lw=1,
               label=r"frozen baseline $\sqrt{2/K}=53\%$")
    ax.set_xlabel("refresh period $T$ (symbols)")
    ax.set_ylabel(r"relative std of $\xi_{\rm block}$")
    ax.set_title("MUI fluctuation shrinks as $1/\\sqrt{F}$")
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.loglog(ts, [r["prf_throughput_bit_per_s"] / 1e6 for r in rows], "o-",
              color="#b9770e", ms=4)
    ax.set_xlabel("refresh period $T$ (symbols)")
    ax.set_ylabel("PRF throughput (Mbit/s)")
    ax.set_title("Cost of refreshing")
    if t_blocked_max is not None:
        for a in axes:
            a.axvline(t_blocked_max, color="#7d3c98", ls="--", lw=0.9,
                      label="last blocked $T$")

    plotting.save(fig, "e12_refresh_period")
    return out


if __name__ == "__main__":
    dump("e12_refresh_period", main())
