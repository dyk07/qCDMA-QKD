"""Timing + correctness probe for the subspace recovery (deleted after use)."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from qcdma_qkd import attack

L, M = 1024, 8
rng = np.random.default_rng(20261110)
C = rng.choice([-1.0, 1.0], size=(M, L))

lines = []
for T in (10, 100, 400, 500, 1000, 2000):
    t0 = time.perf_counter()
    r = attack.code_recovery_from_covariance(
        C, np.ones(M), n_samples=T, chunk=max(T, 1),
        restarts=200, iters=2000, tol=0.05, seed=1)
    dt = time.perf_counter() - t0
    lines.append(f"T={T:5d} {dt:7.2f}s rec={r['recovery_fraction']:.3f} "
                 f"found={r['n_found']:2d} err={r['err_operator_norm']:.1f}")
    print(lines[-1], flush=True)

(ROOT / "tools" / "timing.txt").write_text("\n".join(lines), encoding="utf-8")
