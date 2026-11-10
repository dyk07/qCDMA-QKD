"""Probe: refine the recovery threshold in T (deleted after use)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qcdma_qkd import attack  # noqa: E402

L, M = 1024, 8
rng = np.random.default_rng(20261110)
C = rng.choice([-1.0, 1.0], size=(M, L)).astype(float)
KAPPAS = {"equal": np.ones(M), "weak": np.concatenate([np.ones(M - 1), [0.01]])}

lines = []
for T in (100, 150, 200, 300, 400, 500, 600, 800, 1000):
    parts = []
    for name, kappa in KAPPAS.items():
        best = 0.0
        found = 0
        err = 0.0
        # several seeds: the threshold is statistical, so require robustness
        for sd in (1, 2, 3, 4):
            r = attack.code_recovery_from_covariance(
                C, kappa, n_samples=T, chunk=T, restarts=200, iters=2000,
                tol=0.05, seed=sd)
            best = max(best, r["recovery_fraction"])
            found = max(found, r["n_found"])
            err = r["err_operator_norm"]
        parts.append(f"{name}: rec_max={best:.3f} found={found} err={err:.1f}")
    lines.append(f"T={T:5d}  " + " | ".join(parts))

Path(ROOT / "tools" / "_thr.txt").write_text("\n".join(lines), encoding="utf-8")
print("ok")
