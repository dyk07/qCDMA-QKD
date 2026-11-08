"""q-CDMA-CV-QKD closed-loop modelling package.

Modules
-------
cvqkd      : GG02 (coherent-state) CV-QKD asymptotic / finite-size secret-key rate.
codes      : spreading-code generation (random / Walsh-Hadamard) and cross-correlations.
mui        : dynamic multi-user-interference (MUI) statistics, tail bounds, outage.
budget     : key budget of the closed loop (QKD -> KDF -> auth / seed / application).
security   : composable security-parameter bookkeeping (epsilon composition).
plotting   : shared matplotlib helpers.

All quantities are in shot-noise units (SNU) unless stated otherwise.
"""

from . import cvqkd, codes, mui, budget, security  # noqa: F401

__all__ = ["cvqkd", "codes", "mui", "budget", "security"]
__version__ = "0.1.0"
