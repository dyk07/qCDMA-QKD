"""E12: PRF seed-refresh period T as a three-way trade-off.

The knob
--------
``T`` = symbols between PRF re-draws of the codebook from the seed.  It is the
single parameter that trades

    security      (Eve gets only T samples per codebook)
    MUI stability (F = block/T independent frames stabilise the block average)
    PRF compute   (M*L bits per refresh, i.e. M*L/T bits per symbol)

Method (analytic first, then verify at a few points)
----------------------------------------------------
Every leg is a closed form in ``T`` (``attack.three_way_tradeoff``), so the
full curve costs milliseconds:

    attack : blocked  <=>  T < n_th(L) = 4c^2 (L + ln(2/eps)) / snr_eff^2
    MUI    : rel.std[xi_block] = sqrt(2 sum kappa^2)/sum kappa / sqrt(F),
             F = block/T   ->   ~ sqrt(T/block)
    PRF    : M L / T bits per symbol, i.e. M R_c / T bit/s

Only the *attack* leg has no honest closed form for its transition point, so it
is measured directly at a handful of ``T`` values (multi-seed, because the
transition is statistical).  The concentration constant ``4c^2`` is loose by
roughly two orders of magnitude; this experiment quantifies that gap rather than
hiding it.

Why the previous version ran all night
--------------------------------------
It ran a full seed-recovery simulation at *every* point of a fine ``T`` grid, and
each point was slow for a reason that turned out to be a bug: the ``+-1`` sign
iteration falls into a length-2 limit cycle whenever the subspace contains no
``+-1`` vector -- exactly the case for every *failed* attack -- so every such
call burned the entire iteration budget.  Worse, the cycle detector kept a
Python ``set`` of 1024-element tuples, hashing ~1 KB per iteration.  Comparing a
few arrays with ``np.array_equal`` instead cut a failed call from minutes to
~2 s.

Outputs
-------
results/figures/e12_refresh_period.png
results/data/e12_refresh_period.json
"""

from __future__ import annotations

import numpy as np

from _common import dump, header, scenario

from qcdma_qkd import attack, mui, plotting

BLOCK_SYMBOLS = 10 ** 8
SYMBOL_CHIP_RATE = 1.0e9
RECOVERY_THRESHOLD = 0.99

# Analytic curve: dense in log T, cheap.
T_CURVE = np.logspace(0, 9, 60)

# Measured verification points (the "point check" step).
T_MEASURE = (100, 300, 400, 500, 1000, 10_000)
SEEDS = (1, 2, 3)


def main() -> dict:
    header("E12  PRF refresh period T: security vs MUI stability vs PRF cost")
    sc = scenario()
    cdma = sc["cdma"]
    L = int(cdma.spreading_factor)
    M = int(cdma.n_users)
    kappa = np.ones(M)                    # equal powers: worst case for reading

    # ------------------------------------------------------------------ #
    # 1. Analytic three-way curves (milliseconds)
    # ------------------------------------------------------------------ #
    curve = attack.three_way_tradeoff(L, kappa, T_CURVE,
                                      block_symbols=BLOCK_SYMBOLS,
                                      chip_rate=SYMBOL_CHIP_RATE)
    n_th = curve["n_threshold"]
    print(f"  L={L} M={M}  snr_eff={curve['snr_eff']:.5f}")
    print(f"  analytic threshold n_th = {n_th:,.0f} symbols "
          f"(block = {BLOCK_SYMBOLS:.0e})")

    # PRF hardware cap -> lower bound on T, since PRF(T) = M*Rc/T bit/s
    prf_cap = {f"{cap:.0e}": float(M * SYMBOL_CHIP_RATE / cap)
               for cap in (1e6, 1e7, 1e8)}
    t_secure_max = float(n_th)                       # sufficient bound

    # ------------------------------------------------------------------ #
    # 2. Measured verification at a few T
    # ------------------------------------------------------------------ #
    rng = np.random.default_rng(20261111)
    C = rng.choice([-1.0, 1.0], size=(M, L)).astype(float)

    measured = []
    for T in T_MEASURE:
        fracs, founds = [], []
        for sd in SEEDS:
            r = attack.code_recovery_from_covariance(
                C, kappa, n_samples=int(T), chunk=max(int(T), 1),
                restarts=200, iters=2000, tol=0.05, seed=sd)
            fracs.append(r["recovery_fraction"])
            founds.append(r["n_found"])
        rec_frac = float(max(fracs))      # attack succeeds if ANY seed works
        pred = attack.three_way_tradeoff(L, kappa, [T],
                                         block_symbols=BLOCK_SYMBOLS,
                                         chip_rate=SYMBOL_CHIP_RATE)["rows"][0]
        row = dict(T=float(T), recovery_fraction=rec_frac,
                   recovery_min=float(min(fracs)),
                   n_found=int(max(founds)),
                   recovered=bool(rec_frac >= RECOVERY_THRESHOLD),
                   predicted_blocked=pred["attack_blocked"],
                   margin=pred["margin"],
                   xi_rel_std=pred["xi_rel_std"],
                   prf_bits_per_symbol=pred["prf_bits_per_symbol"])
        measured.append(row)
        print(f"  T={T:>8d}  measured rec={rec_frac:.3f} "
              f"(min {min(fracs):.2f})  found={int(max(founds)):2d}  "
              f"predicted_blocked={pred['attack_blocked']!s:5s}  "
              f"margin={pred['margin']:.4f}")

    blocked = [r["T"] for r in measured if not r["recovered"]]
    t_blocked_max = max(blocked) if blocked else None
    t_success_min = min([r["T"] for r in measured if r["recovered"]], default=None)

    # ------------------------------------------------------------------ #
    # 3. Summary
    # ------------------------------------------------------------------ #
    out = dict(
        L=L, M=M, block_symbols=BLOCK_SYMBOLS,
        symbol_chip_rate=SYMBOL_CHIP_RATE,
        kappa=[float(x) for x in kappa],
        n_threshold=float(n_th),
        snr_eff=float(curve["snr_eff"]),
        xi_mean=float(curve["xi_mean"]),
        xi_rel_std_frozen=float(curve["xi_rel_std_frozen"]),
        curve={k: (v.tolist() if isinstance(v, np.ndarray) else v)
               for k, v in curve.items() if k != "rows"},
        curve_rows=curve["rows"],
        measured=measured,
        t_measured_max_blocked=t_blocked_max,
        t_measured_min_success=t_success_min,
        t_secure_max_sufficient=t_secure_max,
        conservative_gap=(float(t_success_min / n_th)
                          if t_success_min else None),
        prf_period_floor_for_cap=prf_cap,
        note=("security leg covers the second-moment attack only; the MUI and "
              "PRF legs are exact within the stated model"),
    )

    # ------------------------------------------------------------------ #
    # figure
    # ------------------------------------------------------------------ #
    fig, axes = plotting.plt.subplots(1, 3, figsize=(12.8, 3.5))

    ax = axes[0]
    ax.loglog(T_CURVE, curve["margin"], "-", color="#2471a3", lw=1.6,
              label=r"$T/n_{th}$")
    ax.axhline(1.0, color="#7f8c8d", ls=":", lw=1, label=r"$T=n_{th}$")
    succ = [r["T"] for r in measured if r["recovered"]]
    fail = [r["T"] for r in measured if not r["recovered"]]
    if fail:
        ax.loglog(fail, [n_th / t for t in fail], "o", color="#c0392b", ms=7,
                  label="measured: recovered")
    if succ:
        ax.loglog(succ, [n_th / t for t in succ], "x", color="#1e8449", ms=9,
                  label="measured: also recovered")
    ax.set_xlabel("refresh period $T$ (symbols)")
    ax.set_ylabel("needed / available")
    ax.set_title(r"(a) Security: $T < n_{th}$")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.loglog(T_CURVE, curve["xi_rel_std"], "-", color="#1e8449", lw=1.6)
    ax.axhline(curve["xi_rel_std_frozen"], color="#7f8c8d", ls=":", lw=1,
               label=r"frozen: $\sqrt{2/K}$")
    ax.set_xlabel("refresh period $T$ (symbols)")
    ax.set_ylabel(r"rel. std of $\xi_{\rm block}$")
    ax.set_title(r"(b) MUI: $\propto 1/\sqrt{F}$")
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.loglog(T_CURVE, curve["prf_bit_per_s"], "-", color="#b9770e", lw=1.6)
    for cap, tmin in prf_cap.items():
        ax.axhline(float(cap), ls=":", lw=0.9)
        ax.annotate(f"{cap} bit/s (T$\\geq${tmin:,.0f})",
                    (T_CURVE[0], float(cap)), fontsize=6, va="bottom")
    ax.set_xlabel("refresh period $T$ (symbols)")
    ax.set_ylabel("PRF throughput (bit/s)")
    ax.set_title(r"(c) Compute: $\propto 1/T$")

    for a in axes:
        if t_success_min:
            a.axvline(t_success_min, color="#7d3c98", ls="--", lw=1.0)

    plotting.save(fig, "e12_refresh_period")
    return out


if __name__ == "__main__":
    dump("e12_refresh_period", main())
