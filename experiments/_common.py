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


def rate_at(cfg: CVQKDConfig, distance_km: float, extra_xi: float = 0.0,
            n_symbols: int | None = None, **finite_kw) -> dict:
    """Rate ingredients at one distance (eta folded in exactly once).

    ``n_symbols=None`` gives the **asymptotic** rate; passing a block length
    gives the **finite-size** rate, which is what the closed loop uses and what
    the documents argue in.  Use the latter for any number that ends up in a
    conclusion -- the two differ by 23% at ``n = 1e8`` and the finite-size rate
    is zero below ``n ~ 1e7`` at long range.
    """
    T = cvqkd.total_transmittance(distance_km, cfg.alpha_db_per_km, cfg.eta)
    xi = cfg.excess_noise + extra_xi
    if n_symbols is None:
        return cvqkd.asymptotic_rate(cfg.V_A, T, xi, cfg.eta, cfg.v_el,
                                     cfg.beta, cfg.detection)
    return cvqkd.finite_size_rate(cfg.V_A, T, xi, cfg.eta, cfg.v_el, cfg.beta,
                                  cfg.detection, n_symbols=int(n_symbols),
                                  **finite_kw)


def rate_fn(cfg: CVQKDConfig, symbol_rate: float, extra_xi: float = 0.0,
            n_symbols: int | None = None, **finite_kw):
    """Return ``f(distance_km) -> (key_rate_bps, I_AB)`` for budget sweeps.

    ``symbol_rate`` must be the **per-user** symbol rate (``chip_rate / L``),
    not the chip rate.  Passing the chip rate overstates the key rate by L.
    ``n_symbols`` selects the finite-size model (see :func:`rate_at`).
    """
    def f(distance_km: float):
        r = rate_at(cfg, float(distance_km), extra_xi, n_symbols, **finite_kw)
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
