"""Default scenario parameters shared by all experiments.

Every number here is a *modelling* choice, documented in ``docs/``.
Changing a default here changes every experiment, so keep the docs in sync.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CVQKDConfig:
    """Physical layer (GG02, coherent states, reverse reconciliation).

    Validation points (see ``tests/test_cvqkd.py`` and docs/03):
    * ideal limit ``V_A=4, T=1, xi=0, eta=1, v_el=0, beta=1`` gives exactly
      ``0.5*log2(5) = 1.1610`` bits/symbol;
    * ``eta=0.9, v_el=0.05, xi=0.01, beta=0.95`` gives ~0.66 bits/symbol at
      0 km and ~0.029 bits/symbol at 50 km -- the range reported for modern
      homodyne CV-QKD experiments.

    The legacy ``eta=0.6, v_el=0.1`` set is kept available for sensitivity
    analysis (it costs roughly a factor 3 in rate).
    """

    V_A: float = 4.0            # Alice modulation variance (SNU)
    excess_noise: float = 0.01  # channel excess noise, referred to channel input (SNU)
    eta: float = 0.9            # detector efficiency
    v_el: float = 0.05          # electronic noise (SNU, per quadrature, detector output)
    beta: float = 0.95          # reconciliation efficiency
    alpha_db_per_km: float = 0.2
    detection: str = "homodyne"  # homodyne only; see cvqkd.holevo_bound
    # NOTE: the *per-user symbol* rate is NOT an independent parameter here.  It
    # is fixed by the hardware chip rate and the spreading factor:
    #     symbol_rate = chip_rate / L
    # See ``CDMAConfig.symbol_rate()``.  A free ``symbol_rate`` sitting next to
    # ``L`` silently implies impossible chip rates (1e9 sym/s * L=1024 =
    # 1.02 Tchip/s), so it has been removed from this dataclass.


@dataclass
class CDMAConfig:
    """q-CDMA multiplex.

    Rate bookkeeping
    ----------------
    The hardware runs at the **chip** rate ``chip_rate`` (chips/s).  Each user
    transmits one **symbol** per ``L`` chips, so::

        symbol_rate = chip_rate / L                  [symbols/s per user]
        bit_rate    = S [bits/symbol] * symbol_rate  [bits/s per user]

    ``L`` therefore costs a factor ``L`` in rate *in addition to* being the
    parameter that suppresses MUI and (see docs/05) constrains the refresh
    fraction.  Never treat ``symbol_rate`` as a free knob.

    Code model
    ----------
    ``code_model`` selects how the codebook is expanded from the seed, and it
    decides whether a refresh constraint exists at all:

    ``"prf"`` (default, and the choice made for this project's sample model)
        ``c_n = PRF_{c_seed}(...)``.  Observing output bits does **not** leak
        seed bits, so a leaked chip consumes no seed entropy and
        ``f_min = 0``.  **Use ``refresh_fraction = 1.0``** anyway: fresh codes
        every round cost nothing (the PRF expands them locally) and remove
        cross-round correlation entirely.  Security of this layer is
        **computational** (a PRF assumption), not information-theoretic.

    ``"extractor"`` (information-theoretic alternative)
        The seed is fed to a 2-universal extractor, so each known chip is one
        linear constraint on the seed: ``H_min ~ seed_bits - k_known``.  This
        *is* information-theoretic, but it forces ``f >= f_min`` because a
        256-bit seed cannot absorb 1024 known chips.  See
        ``docs/05_security.md`` §5 for the full derivation; the constraint is
        computed by :meth:`refresh_constraint`.

    The sample model uses ``"prf"`` with ``f = 1``.  The extractor path is kept
    fully implemented so the *price* of information-theoretic codes is a
    computable number rather than a hand-wave.
    """

    n_users: int = 8
    spreading_factor: int = 1024      # L, chips per symbol
    chip_rate: float = 1.0e9          # chips / s (hardware line rate)
    code_family: str = "random"       # "random" | "walsh"
    code_model: str = "prf"           # "prf" (computational, default) | "extractor"
    refresh_fraction: float = 1.0     # fraction of chips re-randomised each round

    def __post_init__(self) -> None:
        if self.code_model not in ("prf", "extractor"):
            raise ValueError(
                "code_model must be 'prf' or 'extractor', got %r"
                % (self.code_model,))
        if not 0.0 <= float(self.refresh_fraction) <= 1.0:
            raise ValueError("refresh_fraction must be in [0, 1]")

    def symbol_rate(self) -> float:
        """Per-user symbol rate, ``chip_rate / L`` (symbols/s)."""
        return self.chip_rate / float(self.spreading_factor)

    def bit_rate(self, rate_per_symbol: float) -> float:
        """Convert a bits/symbol figure into bits/s for one user."""
        return float(rate_per_symbol) * self.symbol_rate()

    def symbols_per_second(self) -> float:
        """Alias of :meth:`symbol_rate` (explicit name at call sites)."""
        return self.symbol_rate()

    def is_information_theoretic(self) -> bool:
        """True when the code layer claims information-theoretic security."""
        return self.code_model == "extractor"

    def refresh_constraint(self, seed_bits: int, eps_code: float = 1e-12,
                           p_blind: float = 1.0) -> dict:
        """Refresh constraint implied by ``code_model``.

        Delegates to :mod:`qcdma_qkd.attack` so the formula lives in exactly one
        place.  For ``code_model="prf"`` the returned ``f_min`` is 0; for
        ``code_model="extractor"`` it is the exact binomial-tail bound.
        """
        from . import attack as _attack

        if self.code_model == "prf":
            out = _attack.refresh_security_constraint(
                seed_bits, self.spreading_factor, p_blind,
                eps_code=eps_code, model="prf")
        else:
            out = _attack.refresh_constraint_tail(
                seed_bits, self.spreading_factor, p_blind, eps_code=eps_code)
        out["code_model"] = self.code_model
        out["refresh_fraction"] = float(self.refresh_fraction)
        if self.code_model == "prf":
            out["satisfied"] = True       # no constraint under a PRF
            out["security"] = "computational"
        else:
            out["satisfied"] = bool(
                self.refresh_fraction + 1e-12 >= out["f_min"])
            out["security"] = "information-theoretic (extractor model)"
        return out

    def check_refresh(self, seed_bits: int, eps_code: float = 1e-12,
                      p_blind: float = 1.0) -> dict:
        """Same as :meth:`refresh_constraint` but raises on a violation.

        Use this at the top of a run when the design is supposed to be valid.
        """
        c = self.refresh_constraint(seed_bits, eps_code, p_blind)
        if not c["satisfied"]:
            raise ValueError(
                "refresh_fraction=%g violates the extractor constraint "
                "(f_min=%.4f for seed=%d, L=%d, eps_code=%g). Either raise f "
                "or use code_model='prf'."
                % (self.refresh_fraction, c["f_min"], seed_bits,
                   self.spreading_factor, eps_code))
        return c


@dataclass
class BudgetConfig:
    """Closed-loop key budget."""

    # classical-channel authentication
    mac_model: str = "axu"       # "axu" (bounded-reuse eps-AXU) | "otp" (one-time, IT)
    tag_bits: int = 128
    hash_key_bits: int = 256
    mac_reuse_limit: int = 2 ** 20

    # per-round messages that must be authenticated (bits)
    n_meta_bits: int = 2048      # control metadata per message (frame/sync/seed)
    # Parameter estimation discloses a *fraction* of the symbols, each as a
    # real number.  ``n_pe_bits_per_symbol`` is expressed per *total* symbol:
    #     pe_bits = n_pe_bits_per_symbol * n_symbols
    # Default: 10% of symbols disclosed at 64 bits each -> 6.4 bits/symbol.
    pe_sample_fraction: float = 0.1
    pe_bits_per_real: float = 64.0
    n_pe_bits_per_symbol: float = 6.4
    recon_leak_factor: float = 1.05      # transcript length / I_AB

    # code seed
    seed_bits: int = 256

    # bootstrap
    k0_bits: int = 2 ** 20       # pre-shared authentication root
    k0_reserve_frac: float = 0.1


@dataclass
class SecurityConfig:
    """Security parameters (composable, per component)."""

    eps_qkd: float = 1e-10
    eps_auth: float = 1e-12
    eps_code: float = 1e-12
    eps_mui: float = 1e-12
    eps_pa: float = 1e-12
    eps_sm: float = 1e-10        # smooth-min-entropy parameter
    eps_total: float = 1e-9      # global target


DEFAULTS = dict(
    cvqkd=CVQKDConfig(),
    cdma=CDMAConfig(),
    budget=BudgetConfig(),
    security=SecurityConfig(),
)
