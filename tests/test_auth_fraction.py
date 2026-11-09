"""Is the classical "spend ~10% of the key on authentication" rule still true here?

Answer: **no**, and this module pins down why, so the claim cannot silently drift.

The 10% rule is an artifact of the OTP era: it presumes the authentication key
must be as long as the message, so a transcript several times larger than the
secret key forces a sizeable reservation.  Under a Wegman-Carter / eps-AXU MAC
the key length sets only the *forgeability*, never the message length, so the
cost scales with the number of **messages** instead of their size
(``docs/06_key_budget.md`` §2).

Measured in this project:
    eps-AXU (default): a/K = 3.1e-8
    OTP (worst case) : a/K = 1.98%

Neither is 10%.  The project has NO "reserve 10% for authentication" knob; the
only 0.1 in the config is ``k0_reserve_frac``, a *failure-recovery reserve on
the offline root K_0*, which is a different quantity.

These tests also lock the closed form and the fragmentation threshold at which
10% would actually appear, so that the docs' numbers stay checkable.
"""

from __future__ import annotations

import math

import pytest

from qcdma_qkd import budget, cvqkd, mui
from qcdma_qkd.config import BudgetConfig, CDMAConfig, CVQKDConfig

CFG = CVQKDConfig()
CDMA = CDMAConfig()
L = CDMA.spreading_factor
# 25 km with the eps_mui quantile charged, i.e. the default scenario
T_25 = cvqkd.total_transmittance(25.0, CFG.alpha_db_per_km, CFG.eta)
KAPPA = mui.uniform_kappa(CDMA.n_users - 1, 1.0)
XI_REQ = mui.required_xi(1e-12, KAPPA, L)
RATE_25 = cvqkd.asymptotic_rate(CFG.V_A, T_25, CFG.excess_noise + XI_REQ,
                                CFG.eta, CFG.v_el, CFG.beta)
S_POS = RATE_25["S_pos"]
I_AB = RATE_25["I_AB"]
N_SYMBOLS = 10 ** 8


def _fraction(mac_model: str, n_symbols: int = N_SYMBOLS,
              bits_per_message: int = 1 << 20) -> float:
    cfg = BudgetConfig(mac_model=mac_model)
    b = budget.round_budget(S_POS * n_symbols, n_symbols, I_AB, cfg,
                            bits_per_message=bits_per_message)
    return b["a"] / b["K"]


# --------------------------------------------------------------------------- #
# the project never uses a fixed 10% reservation
# --------------------------------------------------------------------------- #
def test_no_auth_fraction_knob_in_the_config():
    """There must be no 'reserve a fraction for auth' parameter.

    0.1 exists only as k0_reserve_frac (failure recovery on K_0) and
    pe_sample_fraction (10% of symbols disclosed).  Neither is an auth ratio.
    """
    cfg = BudgetConfig()
    names = set(vars(cfg))
    assert "k0_reserve_frac" in names
    assert "pe_sample_fraction" in names
    # no field whose name suggests an authentication share
    for name in names:
        assert not ("auth" in name and "frac" in name), name


def test_pe_fraction_and_k0_reserve_are_not_the_auth_ratio():
    """Both 0.1 fields are unrelated to a/K -- state that in code."""
    cfg = BudgetConfig()
    assert cfg.pe_sample_fraction == 0.1      # parameter-estimation disclosure
    assert cfg.k0_reserve_frac == 0.1         # bootstrap failure reserve
    # the real auth ratio is far below 0.1 (see test_auth_fraction_* for the
    # corrected values: one fresh 208-bit MAC key per round)
    assert _fraction("axu") < 1e-3
    assert _fraction("otp") < 0.05


# --------------------------------------------------------------------------- #
# measured values
# --------------------------------------------------------------------------- #
def test_axu_auth_fraction_is_one_key_per_round():
    """One fresh MAC key per round, so the cost is the *full* l = t+2log2(1/eps).

    An earlier version amortised over ``mac_reuse_limit`` (2^20) while the loop
    re-derived the key every round, understating the cost by ~1572x.  The real
    figure is ~208 bit/round -> a/K = 4.8e-5, still negligible.
    """
    assert _fraction("axu") == pytest.approx(4.82e-5, rel=0.05)


def test_otp_auth_fraction_is_about_two_percent():
    """Even the worst case (fresh key per message) is ~2%, not 10%."""
    assert _fraction("otp") == pytest.approx(0.0198, rel=0.05)


# --------------------------------------------------------------------------- #
# the closed form, and why it is independent of block length
# --------------------------------------------------------------------------- #
def test_closed_form_matches_measurement():
    """a/K ~ l*(pe_per_symbol + factor*I_AB) / (B*S_pos).

    Numerator and denominator are both proportional to n_symbols, so the ratio
    is asymptotically independent of the block length.
    """
    cfg = BudgetConfig(mac_model="otp")
    l = budget.auth_key_bits(cfg)["key_length"]
    pe_per_symbol = ((cfg.n_pe_bits_per_symbol
                      or cfg.pe_sample_fraction * cfg.pe_bits_per_real))
    B = 1 << 20
    predicted = l * (pe_per_symbol + cfg.recon_leak_factor * I_AB) / (B * S_POS)
    assert predicted == pytest.approx(0.0196, rel=0.02)
    # note: OTP has reuse=1, so its per-round cost is n_msg*l and the closed
    # form above (which uses one l per *message*) is the right one
    # matches the simulator at a long block
    assert _fraction("otp", n_symbols=10 ** 9) == pytest.approx(
        predicted, rel=0.02)


def test_auth_fraction_is_blind_to_block_length():
    """n=1e8 vs n=1e9 differ only by the fixed control-message term."""
    small = _fraction("otp", n_symbols=10 ** 7)
    large = _fraction("otp", n_symbols=10 ** 9)
    assert abs(small - large) < 0.005


# --------------------------------------------------------------------------- #
# where 10% would come from: small fragments or long tags
# --------------------------------------------------------------------------- #
def test_ten_percent_requires_much_smaller_fragments():
    """10% appears at ~2e5 bit fragments, 5x below the 1 Mbit default."""
    assert _fraction("otp", bits_per_message=1 << 20) < 0.03
    assert _fraction("otp", bits_per_message=190_000) == pytest.approx(
        0.10, abs=0.02)
    assert _fraction("otp", bits_per_message=10_000) > 1.0   # budget collapses


def test_ten_percent_would_need_a_647_bit_tag():
    """Conversely, keeping 1 Mbit fragments but inflating the tag to ~647 bit."""
    cfg = BudgetConfig(mac_model="otp")
    k_bits = S_POS * N_SYMBOLS
    n_msg = budget.round_message_count(N_SYMBOLS, I_AB, cfg)
    per_message_needed = 0.10 * k_bits / n_msg
    assert per_message_needed == pytest.approx(647, rel=0.05)
    assert budget.auth_key_bits(cfg)["per_message"] == 128.0


def test_axu_ten_percent_is_unreachable():
    """Under eps-AXU, even a huge eps target cannot reach a 10% share."""
    for eps in (1e-12, 1e-6, 1e-3, 0.1):
        per_msg = budget.auth_key_bits(BudgetConfig(mac_model="axu"),
                                       eps_auth=eps)["per_message"]
        assert per_msg * 700 / (S_POS * N_SYMBOLS) < 1e-5


# --------------------------------------------------------------------------- #
# the reusable prior result this rests on
# --------------------------------------------------------------------------- #
def test_wegman_carter_key_length_is_message_independent():
    """l = t + 2 log2(1/eps): no message-length term -- the prior result used.

    This is the whole reason the 10% rule fails: the key does not have to match
    the message.  If a message-length term ever creeps in, this test fires.
    """
    cfg = BudgetConfig(mac_model="axu", tag_bits=128)
    info = budget.auth_key_bits(cfg, eps_auth=1e-12)
    t = info["tag_bits"]
    assert info["key_length"] == pytest.approx(t + 2 * math.log2(1 / 1e-12))
    # 500 messages are still served by ONE key, so the cost saturates at l
    assert budget.auth_bits(500, cfg) == pytest.approx(info["key_length"])
    assert budget.auth_bits(50_000, cfg) == pytest.approx(info["key_length"])
