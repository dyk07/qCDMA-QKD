# qCDMA-QKD

A research implementation of the **code-distribution loop** for q-CDMA-CV-QKD.
It turns the trust gap around the secure distribution and refresh of CDMA codes
into an object that can be computed, reduced, and budgeted.

**This repository does not claim to improve the quantum-security lower bound of
CV-QKD.** See the claim boundary in [`docs/00_summary.md`](docs/00_summary.md).

## Quick Start

```bash
pip install numpy scipy matplotlib pytest

python experiments/run_all.py    # E0-E13, about 2 minutes; generates JSON
python -m pytest                 # 360 tests: physics, statistics, units, and docs
python tools/doc_check.py        # Check that docs/ numbers match results/data/
```

Outputs:
- `results/figures/*.png` - figures from the experiments
- `results/data/*.json` - all numerical results (the **single source of truth**
  for documented numbers; rerun `run_all.py` after changing code)

### Why `tools/doc_check.py` Exists

This project has three times encountered the situation where the code was
correct, the documented numbers were stale, and all tests still passed:
authentication cost remained at the amortized value `0.132 bit` instead of the
actual `207.7 bit` per round, the loop remained at the asymptotic rate instead
of the finite-blocklength value (`4.31` -> `3.33` Mbit), and $\epsilon$
composition still referenced an obsolete round limit. **Unit tests cannot see
prose.**

This script treats `results/data/*.json` as the sole source of truth and checks
that:
1. every key number in the documentation can be recomputed from JSON
   (`CLAIMS`);
2. retired values do not reappear (`RETIRED`) unless the line explicitly
   discusses historical values.

`tests/test_docs_consistency.py` integrates both checks into `pytest`, so
documentation drift **turns CI red** instead of waiting for a reader to find
it.

## Repository Layout

```
qCDMA-QKD/
├── tools/
│   └── doc_check.py            Documentation ↔ results/data consistency checks
├── docs/                       Research documentation (README.md is the index)
│   ├── initial.md              Original paper outline (requirements baseline)
│   ├── 00_summary.md           Core conclusions and key numbers
│   ├── 01_model_and_assumptions.md   Threat model, assumptions, notation, units
│   ├── 02_protocol.md          The code-distribution loop
│   ├── 03_cvqkd_model.md       CV-QKD rate model and validation
│   ├── 04_mui_analysis.md      Dynamic MUI statistics (core physics correction)
│   ├── 05_security.md          Security reductions, leakage, refresh constraints
│   ├── 06_key_budget.md        Key budget and K_0
│   ├── 07_results.md           E0-E13 results and interpretation
│   ├── 08_open_problems.md     Gaps, limitations, and future work
│   ├── 09_model_vs_bound.md    Model / exact / strict-bound classification
│   ├── 10_parameters.md        Parameter table and the feasible region
│   ├── 11_conclusions.md       Claim boundaries and key numbers for drafting
│   └── paper/                  Manuscripts (REVTeX sources and compiled PDFs)
│       ├── paper_en.tex        Regular Article (English, 21 pages)
│       ├── paper.tex           Chinese version (15 pages)
│       └── letter.tex          Companion Letter (5 pages)
├── src/qcdma_qkd/              Core library
│   ├── config.py               Scenario parameters (chip_rate / L / seeds)
│   ├── cvqkd.py                GG02 rate, Holevo information, finite blocklength, PE
│   ├── codes.py                Spreading-code generation and cross-correlation
│   ├── mui.py                  MUI statistics, tails, and block-average xi
│   ├── budget.py               Authentication, seed, application budgets, K_0
│   ├── security.py             Epsilon composition, LHL, and PRF
│   ├── attack.py               MitM, blind estimation, leakage, refresh constraints
│   ├── loop.py                 Loop simulation (KDF + PRF spreading + rolling auth)
│   └── plotting.py             Consistent plotting style
├── experiments/                E0-E13
├── tests/                      Validation tests
└── results/                    Figures and data
```

## Four Quantitative Findings

| # | Finding | Number |
|---|---|---|
| 1 | Block-average MUI fluctuations **do not decay with block length** | Relative std. $=\sqrt{2/K}$, **53%** for 8 users |
| 2 | Authentication cost is **independent of message length** | 667 messages/round -> **0.13 bit**/round |
| 3 | The refresh constraint is based on Eve's **known** chips and an $\epsilon_{\text{code}}$ target; **extractor branch only** | $f\ge0.7886$ (not the intuitive $0.25$) |
| 4 | The **quantum-layer** value of code secrecy is negligible | Rate loss **<1%** at $L{=}4096$ |

## Code-Layer Security Model: PRF + Full Refresh (**Final Design**)

```python
CDMAConfig(code_model="prf", refresh_fraction=1.0)   # default
```

| Item | PRF + $f=1$ (adopted) | Information-theoretic extractor (alternative) |
|---|---|---|
| Code-layer security | **Computational** (PRF assumption) | Information-theoretic |
| Refresh requirement | None ($f_{\min}=0$) | $f\ge0.7886$ |
| Cross-round correlation | None | Retains $n_{\text{reused}}$ chips per round |
| Additional assumption | PRF | None |

The PRF expands the seed **locally**, so a fresh code every round has zero
communication cost (only computational cost). With $f=1$, cross-round
correlation is deterministically removed (measured
$\langle c_n,c_{n-1}\rangle/L=0.0115\approx O(1/\sqrt L)$).

**The trade-off must be explicit**: PRF provides only **computational** security.
If information-theoretic code-layer security is required, a 256-bit seed cannot
cover the linear equations of $L=1024$ known chips, so $f\ge0.7886$ is required.
The derivation in §5 of `docs/05_security.md` quantifies this trade-off.
Switch with `CDMAConfig(code_model="extractor")`; `check_refresh()` raises on
violations.

## Rates and Units (Common Source of Errors)

The per-user rate is **not** a free parameter; it is determined by the hardware
chip rate and spreading factor:

$$R_s=\frac{R_c}{L}=\frac{10^9}{1024}=0.977\ \text{Msymbol/s}$$

With 1 Gchip/s and $L=1024$, the per-user rate at 25 km for 8 users,
including MUI, is about **42 kbit/s** (not Mbit/s). The rate pays another
factor of $L$.

$L$ appears in three places with different effects, yielding an **optimal
spreading factor** $L^\ast$:

| Effect | As $L$ increases | Direction |
|---|---|---|
| MUI excess noise $\xi\propto 1/L$ | ↓ | Larger is better |
| Symbol rate $R_s=R_c/L$ | ↓ | Larger is worse |
| Refresh constraint $f_{\min}$ (**extractor branch only**) | ↑ | Larger is worse |

The competition between the first two effects gives
$L^\ast\approx1024$-2048 (see E9).

## Common Pitfalls

Several easy-to-reverse details are covered by the tests:

0. **The code-layer security type must be explicit** - PRF (computational
   security, no refresh constraint) versus an information-theoretic extractor
   (with an $f_{\min}$ constraint). The example model uses PRF; mixing the two
   conclusions is a common error.
1. **Authentication cost is counted per message, not per bit** - the MAC key
   does not grow with message length.
2. **The refresh constraint uses Eve's *known* chips** -
   $f\ge1-(\text{seed}-k_{\text{req}})/(Lp)$, not "new chips > seed length"
   (which gives the wrong direction, $f>0.25$). The criterion also cannot be
   written as $H_{\min}>0$ (which permits $\epsilon_{\text{code}}=0.5$);
   $\epsilon_{\text{code}}^{*}$ must be explicit, with
   $k_{\text{req}}=\log_2(1/\epsilon^{*})$. If
   $\text{seed}>Lp+k_{\text{req}}$, the constraint disappears
   ($f_{\min}=0$); this is **not infeasible**. The only truly infeasible region
   is $\text{seed}\le k_{\text{req}}=\log_2(1/\epsilon_{code})$.
3. **`symbol_rate` must be derived from `chip_rate/L`** - setting it freely
   implies impossible Tchip/s-scale hardware and overestimates the rate by $L$.

## Environment

Python >= 3.9, `numpy`, `scipy`, `matplotlib`, and `pytest`. No network
dependencies are required; all calculations are closed-form.
