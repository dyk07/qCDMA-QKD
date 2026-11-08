"""Key budget of the closed q-CDMA loop.

Per round n the QKD block yields ``K_n`` secret bits, which the KDF splits into
three independent, single-use streams:

    K_n  ->  a_n  (authentication key for round n+1 control/classical traffic)
          ->  c_n  (code seed, expanded by a PRF into the spreading codebook)
          ->  u_n  (application output)

Why the auth cost does *not* scale with the number of bits
----------------------------------------------------------
A Wegman-Carter MAC authenticates an arbitrarily long message with a short key:
the key only has to be long enough to make the tag unforgeable, not as long as
the message.  The two regimes we model are

``otp`` (information-theoretic, fresh key per message)
    eps = 2^{-t}, key = t bits per message.  Reusing the key for m messages
    costs t*c bits in total, where c is the *number of MAC invocations* (one per
    classical message), never the message length in bits.

``axu`` (eps-AXU hash, one key reused r times)
    eps <= r * 2^{-t} + 2^{-(l-t)/2}
    => l = t + 2*log2(1/eps),  amortised cost per message = l / r bits.

So the dominant term is the *number of authenticated messages*, which is why
the closed loop is cheap: a round needs O(10) messages (code seed, sync, frame
control, PE commitments, reconciliation-coordinates hashes), not O(10^8).

``budget.round_budget`` therefore takes ``n_messages`` (count), not ``n_bits``.
A helper ``messages_for_traffic`` converts a transcript volume into a message
count under an explicit fragmentation rule so the assumption stays visible.
"""

from __future__ import annotations

import math

import numpy as np

from .config import BudgetConfig


# --------------------------------------------------------------------------- #
# authentication
# --------------------------------------------------------------------------- #
def auth_key_bits(cfg: BudgetConfig, eps_auth: float = 1e-12) -> dict:
    """Key material for one authenticated message in the chosen MAC model."""
    t = float(cfg.tag_bits)
    if cfg.mac_model == "otp":
        per_msg = t
        key_len = t
        reuse = 1
    elif cfg.mac_model == "axu":
        key_len = t + 2.0 * math.log2(1.0 / eps_auth)
        # reuse must not erode the tag term: r * 2^{-t} <= eps_auth
        max_reuse = max(1, int(math.floor(eps_auth * 2.0 ** t)))
        reuse = int(min(cfg.mac_reuse_limit, max_reuse))
        per_msg = key_len / reuse
    else:
        raise ValueError(f"unknown mac_model {cfg.mac_model!r}")
    return dict(per_message=float(per_msg), key_length=float(key_len),
                tag_bits=t, reuse=int(reuse))


def auth_bits(n_messages: int, cfg: BudgetConfig,
              eps_auth: float = 1e-12) -> float:
    """Total authentication-key bits for ``n_messages`` messages."""
    return float(max(int(n_messages), 0) * auth_key_bits(cfg, eps_auth)["per_message"])


def messages_for_traffic(total_bits: float, bits_per_message: int) -> int:
    """Fragmentation rule: one MAC per ``bits_per_message`` bits of transcript."""
    if bits_per_message <= 0:
        raise ValueError("bits_per_message must be positive")
    return int(math.ceil(max(float(total_bits), 0.0) / bits_per_message))


# --------------------------------------------------------------------------- #
# per-round traffic
# --------------------------------------------------------------------------- #
def round_traffic(n_symbols: int, i_ab_bits_per_symbol: float,
                  cfg: BudgetConfig, n_control_messages: int = 8) -> dict:
    """Classical transcript volume per round (bits), split by purpose.

    * ``pe``    : disclosed parameter-estimation data.  Only a *fraction* of the
      symbols is disclosed, and each disclosed real costs ``pe_bits_per_real``
      bits, so the per-symbol cost is
      ``pe_sample_fraction * pe_bits_per_real`` (default 0.1 * 64 = 6.4).
      The default field ``n_pe_bits_per_symbol`` is kept as an override for
      scenarios that quote the value directly.
    * ``recon`` : reconciliation transcript, ``factor * I_AB * n`` bits
    * ``meta``  : control metadata (frame/sync/code-update seed, bits)
    """
    pe_per_symbol = (float(cfg.n_pe_bits_per_symbol)
                     if cfg.n_pe_bits_per_symbol
                     else float(cfg.pe_sample_fraction) * float(cfg.pe_bits_per_real))
    pe = pe_per_symbol * n_symbols
    recon = float(cfg.recon_leak_factor) * float(i_ab_bits_per_symbol) * n_symbols
    meta = float(cfg.n_meta_bits) * n_control_messages
    return dict(pe=pe, recon=recon, meta=meta, total=pe + recon + meta,
                n_control_messages=int(n_control_messages),
                pe_per_symbol=pe_per_symbol)


def round_message_count(n_symbols: int, i_ab_bits_per_symbol: float,
                        cfg: BudgetConfig, bits_per_message: int = 1 << 20,
                        n_control_messages: int = 8) -> int:
    """Number of authenticated messages in a round.

    Control messages are counted individually; the bulk transcript is
    fragmented into ``bits_per_message`` chunks.
    """
    tr = round_traffic(n_symbols, i_ab_bits_per_symbol, cfg, n_control_messages)
    bulk = messages_for_traffic(tr["pe"] + tr["recon"], bits_per_message)
    return int(bulk + n_control_messages)


def round_auth_bits(n_symbols: int, i_ab_bits_per_symbol: float,
                    cfg: BudgetConfig, bits_per_message: int = 1 << 20,
                    n_control_messages: int = 8,
                    eps_auth: float = 1e-12) -> float:
    """Authentication-key bits consumed by one round."""
    n_msg = round_message_count(n_symbols, i_ab_bits_per_symbol, cfg,
                                bits_per_message, n_control_messages)
    return auth_bits(n_msg, cfg, eps_auth)


# --------------------------------------------------------------------------- #
# budget
# --------------------------------------------------------------------------- #
def round_budget(k_bits: float, n_symbols: int, i_ab_bits_per_symbol: float,
                 cfg: BudgetConfig, bits_per_message: int = 1 << 20,
                 n_control_messages: int = 8,
                 eps_auth: float = 1e-12) -> dict:
    """Split ``k_bits`` into a / c / u and report feasibility."""
    n_msg = round_message_count(n_symbols, i_ab_bits_per_symbol, cfg,
                                bits_per_message, n_control_messages)
    a = auth_bits(n_msg, cfg, eps_auth)
    c = float(cfg.seed_bits)
    u = float(k_bits) - a - c
    return dict(K=float(k_bits), a=float(a), c=c, u=u, n_messages=int(n_msg),
                surplus=(u / k_bits if k_bits > 0 else -np.inf),
                feasible=bool(u >= 0),
                traffic=round_traffic(n_symbols, i_ab_bits_per_symbol, cfg,
                                      n_control_messages))


def round_budget_from_rate(key_rate_bps: float, round_seconds: float,
                           n_symbols: int, i_ab_bits_per_symbol: float,
                           cfg: BudgetConfig, **kw) -> dict:
    """Convenience wrapper: key bits available in a round of ``round_seconds``."""
    return round_budget(key_rate_bps * round_seconds, n_symbols,
                        i_ab_bits_per_symbol, cfg, **kw)


def net_rate_bits_per_second(key_rate_bps: float, n_symbols: int,
                             i_ab_bits_per_symbol: float, cfg: BudgetConfig,
                             symbol_rate: float, **kw) -> dict:
    """Application rate after paying auth and seed for one round.

    ``symbol_rate`` is the **per-user** symbol rate, ``chip_rate / L`` -- see
    ``CDMAConfig.symbol_rate()``.  ``key_rate_bps`` must therefore already be a
    per-user figure (``S_pos * symbol_rate``).
    """
    round_seconds = n_symbols / symbol_rate
    b = round_budget(key_rate_bps * round_seconds, n_symbols,
                     i_ab_bits_per_symbol, cfg, **kw)
    return dict(rate_net=max(b["u"], 0.0) / round_seconds,
                rate_raw=key_rate_bps, round_seconds=round_seconds, **b)


def min_key_rate_bps(n_symbols: int, i_ab_bits_per_symbol: float,
                     cfg: BudgetConfig, symbol_rate: float, **kw) -> float:
    """Raw per-user QKD rate below which u_n < 0 (the loop runs at a loss)."""
    a = round_auth_bits(n_symbols, i_ab_bits_per_symbol, cfg, **kw)
    return float((a + cfg.seed_bits) * symbol_rate / n_symbols)


def overhead_ratio(n_symbols: int, i_ab_bits_per_symbol: float, k_bits: float,
                   cfg: BudgetConfig, **kw) -> float:
    """Share of the round key spent on auth + seed."""
    b = round_budget(k_bits, n_symbols, i_ab_bits_per_symbol, cfg, **kw)
    return float((b["a"] + b["c"]) / k_bits) if k_bits > 0 else np.inf


# --------------------------------------------------------------------------- #
# bootstrap (round 0)
# --------------------------------------------------------------------------- #
def k0_budget(cfg: BudgetConfig, n_symbols: int, i_ab_bits_per_symbol: float,
              bootstrap_rounds: int = 1, **kw) -> dict:
    """Size the pre-shared authentication root K_0.

    Round 1 must be authenticated *before* round-1 QKD keys exist, so K_0 must
    cover, for each bootstrap round:

    * the authentication cost ``a`` of that round (its transcript), and
    * the code **seed** ``c`` for that round, because the round-1 codebook
      cannot be derived from a key that does not exist yet.

    Without the seed term the bootstrap is circular: authenticating the
    code-update message requires a code to have been distributed, which requires
    the seed.  A failure-recovery reserve is added on top.
    """
    per_round_a = round_auth_bits(n_symbols, i_ab_bits_per_symbol, cfg, **kw)
    per_round = per_round_a + float(cfg.seed_bits)
    total = per_round * bootstrap_rounds
    reserve = cfg.k0_reserve_frac * total
    return dict(per_round=per_round, per_round_auth=per_round_a,
                per_round_seed=float(cfg.seed_bits),
                total=total, reserve=reserve,
                required=total + reserve, available=float(cfg.k0_bits),
                feasible=bool(cfg.k0_bits >= total + reserve),
                fraction_of_k0=(total + reserve) / cfg.k0_bits)


def k0_lifetime_rounds(cfg: BudgetConfig, n_symbols: int,
                       i_ab_bits_per_symbol: float, **kw) -> float:
    """Rounds a given K_0 can authenticate at the bootstrap cost."""
    per_round = (round_auth_bits(n_symbols, i_ab_bits_per_symbol, cfg, **kw)
                 + float(cfg.seed_bits))
    return float(cfg.k0_bits / per_round) if per_round > 0 else np.inf


# --------------------------------------------------------------------------- #
# seed / code update
# --------------------------------------------------------------------------- #
def seed_rounds_per_k0(k0_bits: int, seed_bits: int) -> float:
    """How many code updates the whole bootstrap key could buy (sanity check)."""
    return float(k0_bits / seed_bits) if seed_bits > 0 else np.inf


def frames_per_seed(L: int, symbol_rate: float, frame_seconds: float,
                    prf_throughput_bps: float | None = None) -> float:
    """Frames of the *same* round one seed can cover.

    ``symbol_rate`` is per-user (``chip_rate / L``); the product ``L *
    symbol_rate`` recovers the hardware chip rate, so ``chips_per_frame`` is
    correct for either convention.  PRF expansion is computationally bounded,
    not key-bounded, so this is limited by throughput (if given) rather than by
    the key budget.
    """
    chips_per_frame = L * frame_seconds * symbol_rate
    if prf_throughput_bps is None:
        return np.inf
    return float(prf_throughput_bps / max(chips_per_frame, 1e-30))


def update_interval_bound(seed_bits: float, key_rate_bps: float,
                          safety_factor: float = 10.0) -> float:
    """Minimum seconds between code updates.

    A new seed costs ``seed_bits``; we require the seed to be at most
    ``1/safety_factor`` of the key produced in that interval.
    """
    if key_rate_bps <= 0:
        return np.inf
    return float(seed_bits * safety_factor / key_rate_bps)


# --------------------------------------------------------------------------- #
# sweeps
# --------------------------------------------------------------------------- #
def budget_vs_distance(distances_km: np.ndarray, rate_fn, cfg: BudgetConfig,
                       n_symbols: int, symbol_rate: float, **kw) -> dict:
    """Raw vs net rate across distance.

    ``rate_fn(L_km) -> (key_rate_bps, i_ab_bits_per_symbol)``
    """
    raw, net, auth, seed, feas = [], [], [], [], []
    for L in np.atleast_1d(distances_km):
        kr, iab = rate_fn(float(L))
        r = net_rate_bits_per_second(kr, n_symbols, iab, cfg, symbol_rate, **kw)
        raw.append(kr)
        net.append(r["rate_net"])
        auth.append(r["a"] / r["round_seconds"])
        seed.append(r["c"] / r["round_seconds"])
        feas.append(r["feasible"])
    return dict(distance=np.asarray(distances_km, dtype=float),
                raw=np.asarray(raw), net=np.asarray(net),
                auth_rate=np.asarray(auth), seed_rate=np.asarray(seed),
                feasible=np.asarray(feas))
