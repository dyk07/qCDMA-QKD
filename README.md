# qCDMA-QKD

A research implementation of the **code-distribution loop** for q-CDMA-CV-QKD:
the secure distribution and refresh of CDMA codes, treated as an object that can
be computed, reduced, and budgeted.

This repository contains the library, the experiment scripts and their numerical
output. The research notes and the manuscripts are **not** published here.

## Quick Start

```bash
pip install numpy scipy matplotlib pytest

python experiments/run_all.py    # E0-E14, about 2 minutes; writes results/
python -m pytest                 # physics, statistics, units, and doc checks
```

Outputs:

- `results/figures/*.png` - figures from the experiments
- `results/data/*.json` - all numerical results (the **single source of truth**;
  rerun `run_all.py` after changing code)

## Repository Layout

```
qCDMA-QKD/
|-- src/qcdma_qkd/              Core library
|   |-- config.py               Scenario parameters (chip_rate / L / seeds)
|   |-- cvqkd.py                GG02 rate, Holevo information, finite blocklength, PE
|   |-- codes.py                Spreading-code generation and cross-correlation
|   |-- mui.py                  MUI statistics, tails, and block-average xi
|   |-- budget.py               Authentication, seed, application budgets, K_0
|   |-- security.py             Epsilon composition, LHL, and PRF
|   |-- attack.py               MitM, blind estimation, leakage, refresh constraints
|   |-- loop.py                 Loop simulation (KDF + PRF spreading + rolling auth)
|   \-- plotting.py             Consistent plotting style
|-- experiments/                E0-E14, driven by run_all.py
|-- tests/                      Validation tests
|-- tools/                      Developer utilities (see below)
\-- results/                    Figures and data
```

## Code-Layer Security: Two Models

`CDMAConfig.code_model` selects how the codebook is expanded from the seed, and
it decides whether a refresh constraint exists at all.

```python
CDMAConfig(code_model="prf", refresh_fraction=1.0)        # default
CDMAConfig(code_model="extractor", refresh_fraction=0.8)  # information-theoretic
```

- `"prf"` - `c_n = PRF_{c_seed}(...)`. Observing output bits does not leak seed
  bits, so `f_min = 0`. Security of this layer is **computational** (a PRF
  assumption). `refresh_fraction = 1.0` is still recommended: fresh codes every
  round cost nothing beyond computation and remove cross-round correlation.
- `"extractor"` - the seed is fed to a 2-universal extractor, so each known chip
  is one linear constraint on the seed. This is **information-theoretic**, but it
  forces `f >= f_min` because a short seed cannot absorb `L` known chips.

Both paths are fully implemented, so the price of the information-theoretic
option is a computable number rather than a hand-wave. `check_refresh()` returns
the constraint and raises on a violation; `refresh_constraint()` returns it
without raising.

## Rates and Units

The per-user rate is **not** a free parameter; it is fixed by the hardware chip
rate and the spreading factor:

```python
symbol_rate = chip_rate / L      # symbols/s per user
bit_rate    = S * symbol_rate    # bits/s per user
```

Never set `symbol_rate` by hand: a freely chosen value implies impossible
Tchip/s-scale hardware and overestimates the rate by a factor `L`.

`L` appears in three places with different signs, which is why an optimal
spreading factor exists:

| Effect | As `L` increases |
|---|---|
| MUI excess noise (proportional to `1/L`) | decreases - larger is better |
| Symbol rate (`R_c / L`) | decreases - larger is worse |
| Refresh constraint `f_min` (`extractor` model only) | increases - larger is worse |

## Common Pitfalls

Several easy-to-reverse details are pinned by tests:

0. **The code-layer security type must be explicit** - PRF (computational, no
   refresh constraint) versus an extractor (information-theoretic, with an
   `f_min` constraint). Mixing the two conclusions is a common error.
1. **Authentication cost is counted per message, not per bit** - the MAC key
   does not grow with message length.
2. **The refresh constraint counts the chips Eve can know**, not "new chips
   exceeding the seed length" (which gives the wrong direction). The criterion
   also cannot be written as `H_min > 0`, which would permit
   `eps_code = 0.5`; `eps_code` must be explicit, with
   `k_req = log2(1/eps_code)`. If `seed > L*p + k_req` the constraint disappears
   (`f_min = 0`), which is *not* infeasible; the only truly infeasible region is
   `seed <= k_req`.
3. **`symbol_rate` must be derived from `chip_rate / L`** - see above.

## Developer Utilities

| Script | Purpose |
|---|---|
| `tools/doc_check.py` | Checks that documented numbers still match `results/data/*.json` |
| `tools/core_workflow.py` | Generates the workflow diagram under `results/figures/` |
| `tools/*_paper.py`, `tools/letter_figure.py` | Print-sized figures for the manuscripts (they write into the unpublished `docs/paper/`) |

`tools/doc_check.py` treats `results/data/*.json` as the sole source of truth:
`CLAIMS` recomputes each documented number from the JSON, `RETIRED` forbids
superseded values from reappearing, and `AUDIT` requires every "number + unit" in
the prose to be either a registered claim or an allowlisted input. **Unit tests
cannot see prose**, so this check exists to catch code-and-prose drift.

The prose it guards is not part of this repository. When `docs/` is absent the
script reports that there is nothing to check and
`tests/test_docs_consistency.py` skips, so a clone of this repository runs green
without it. If you keep the notes locally under `docs/`, the check - and the
corresponding tests - activate automatically.

```bash
python tools/doc_check.py           # human-readable report
python tools/doc_check.py -q        # failures only
python tools/doc_check.py --update  # refresh the AUDIT baseline: review the diff
```

## Environment

Python >= 3.9 with `numpy`, `scipy`, `matplotlib` and `pytest`. No network
dependencies: all calculations are closed-form.
