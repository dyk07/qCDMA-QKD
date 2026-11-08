"""Shared experiment plumbing: scenario construction and result dump."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qcdma_qkd import cvqkd, plotting  # noqa: E402
from qcdma_qkd.config import (BudgetConfig, CDMAConfig, CVQKDConfig,  # noqa: E402
                              SecurityConfig)


def scenario():
    """Default scenario used by every experiment (documented in docs/01)."""
    return dict(cv=CVQKDConfig(), cdma=CDMAConfig(), budget=BudgetConfig(),
                sec=SecurityConfig())


def rate_at(cfg: CVQKDConfig, distance_km: float, extra_xi: float = 0.0) -> dict:
    """Asymptotic rate ingredients at one distance (eta folded in exactly once)."""
    T = cvqkd.total_transmittance(distance_km, cfg.alpha_db_per_km, cfg.eta)
    return cvqkd.asymptotic_rate(cfg.V_A, T, cfg.excess_noise + extra_xi,
                                 cfg.eta, cfg.v_el, cfg.beta, cfg.detection)


def rate_fn(cfg: CVQKDConfig, symbol_rate: float, extra_xi: float = 0.0):
    """Return ``f(distance_km) -> (key_rate_bps, I_AB)`` for budget sweeps.

    ``symbol_rate`` must be the **per-user** symbol rate (``chip_rate / L``),
    not the chip rate.  Passing the chip rate overstates the key rate by L.
    """
    def f(distance_km: float):
        r = rate_at(cfg, float(distance_km), extra_xi)
        return r["S_pos"] * symbol_rate, r["I_AB"]
    return f


def dump(name: str, payload: dict) -> Path:
    """Write a JSON result file into ``results/data``."""
    plotting.setup()
    path = plotting.DATA_DIR / f"{name}.json"

    def _conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(f"not serialisable: {type(o)}")

    path.write_text(json.dumps(payload, indent=2, default=_conv),
                    encoding="utf-8")
    print(f"  data   -> {path.relative_to(ROOT)}")
    return path


def header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
