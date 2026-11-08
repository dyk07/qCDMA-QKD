"""Rate-unit conventions.

The project had a real bug class here: a free ``symbol_rate`` living next to the
spreading factor ``L`` implies ``chip_rate = symbol_rate * L``, which silently
produced impossible hardware (1e9 symbol/s * L=1024 = 1.02 Tchip/s) and
overstated every key rate by a factor of L.

These tests pin the convention:
    symbol_rate = chip_rate / L          (per user)
    bit_rate    = S[bits/symbol] * symbol_rate
"""

from __future__ import annotations

import pytest

from qcdma_qkd.config import CDMAConfig


def test_symbol_rate_is_chip_rate_over_L():
    for L in (64, 256, 1024, 4096):
        c = CDMAConfig(chip_rate=1e9, spreading_factor=L)
        assert c.symbol_rate() == pytest.approx(1e9 / L)


def test_implied_chip_rate_round_trips():
    """L * symbol_rate must recover the hardware chip rate exactly."""
    c = CDMAConfig(chip_rate=2.5e9, spreading_factor=512)
    assert c.spreading_factor * c.symbol_rate() == pytest.approx(c.chip_rate)


def test_default_chip_rate_is_physically_plausible():
    c = CDMAConfig()
    assert c.chip_rate <= 1e11, "chip rate must be a realistic line rate"
    # the per-user symbol rate must be *below* the chip rate by exactly L
    assert c.symbol_rate() == pytest.approx(c.chip_rate / c.spreading_factor)


def test_bit_rate_conversion():
    c = CDMAConfig(chip_rate=1e9, spreading_factor=1024)
    # 0.0431 bits/symbol * (1e9/1024) symbols/s
    assert c.bit_rate(0.0431) == pytest.approx(0.0431 * 1e9 / 1024)


def test_spreading_factor_costs_a_factor_L_in_rate():
    """Doubling L halves the bit rate for the same bits/symbol."""
    a = CDMAConfig(chip_rate=1e9, spreading_factor=512)
    b = CDMAConfig(chip_rate=1e9, spreading_factor=1024)
    assert a.bit_rate(0.5) == pytest.approx(2 * b.bit_rate(0.5))


def test_no_free_symbol_rate_on_the_config():
    """``CVQKDConfig`` must not carry an independent symbol_rate field."""
    from qcdma_qkd.config import CVQKDConfig

    cfg = CVQKDConfig()
    assert not hasattr(cfg, "symbol_rate"), (
        "a free symbol_rate next to L re-introduces the Tchip/s bug; "
        "use CDMAConfig.symbol_rate()"
    )
