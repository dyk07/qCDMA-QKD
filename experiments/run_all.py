"""Run every experiment in order.

Usage
-----
    python experiments/run_all.py            # all experiments
    python experiments/run_all.py e2 e4      # a subset

Each experiment is a module in ``experiments/`` exposing ``main() -> dict`` and
writing to ``results/{figures,data}``.
"""

from __future__ import annotations

import importlib
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

EXPERIMENTS = ["e0_headline", "e1_cvqkd_baseline", "e2_mui", "e3_mitm_auth", "e4_refresh",
               "e5_budget", "e6_epsilon", "e7_loop", "e8_state_prep",
               "e9_spreading_tradeoff", "e10_model_vs_bound",
               "e11_covariance_attack", "e12_refresh_period",
               "e13_kappa_concentration", "e14_feasibility"]


def run(names: list[str]) -> int:
    from _common import dump

    failures = 0
    for name in names:
        mod_name = next((m for m in EXPERIMENTS if m.startswith(name)), None)
        if mod_name is None:
            print(f"!! unknown experiment {name!r}")
            failures += 1
            continue
        t0 = time.perf_counter()
        try:
            mod = importlib.import_module(mod_name)
            payload = mod.main()
            dump(mod_name, payload)
        except Exception:
            failures += 1
            print(f"!! {mod_name} failed")
            traceback.print_exc()
        else:
            print(f"   {mod_name} ok in {time.perf_counter() - t0:.1f}s")
    return failures


if __name__ == "__main__":
    args = sys.argv[1:]
    names = args if args else EXPERIMENTS
    print(f"running {len(names)} experiment(s)")
    sys.exit(1 if run(names) else 0)
