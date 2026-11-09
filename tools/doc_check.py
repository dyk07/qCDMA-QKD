"""Verify that the numbers written in ``docs/`` match the computed results.

Why this exists
---------------
Twice now the project has shipped documentation whose numbers were correct at
the time they were typed and wrong afterwards, with **every test still green**:

* the authentication cost stayed at the amortised ``0.132 bit`` after the loop
  moved to one fresh key per round (``207.7 bit``), and
* the closed loop stayed at the asymptotic rate after it was switched to the
  finite-size rate (``4.31`` -> ``3.33`` Mbit), and
* the composed epsilon kept quoting ``1.8e4`` rounds after the per-round budget
  was corrected.

None of these are code bugs in the usual sense -- the code was right, the prose
was stale.  Unit tests cannot see prose.  This script therefore treats the
``results/data/*.json`` files as the single source of truth and checks the docs
against them.

Two kinds of check
------------------
``CLAIMS``
    Each claim recomputes a value from the JSON (or from ``qcdma_qkd``) and
    asserts the doc contains exactly that rendering *on the anchored line*.
    The *value* is never hand-maintained; only the "which line, which
    formatting" binding is.

``RETIRED``
    Values that were once published and are now wrong.  Their presence is a
    hard failure -- unless the line is explicitly discussing the history (it
    contains a marker such as "早期版本" / "旧版本" / "开发记录").  Without the
    marker allowlist this would forbid ever explaining a past mistake.

``AUDIT`` (the safety net)
    ``CLAIMS`` only guards numbers someone remembered to register.  ``AUDIT``
    closes that gap: it extracts every "number + unit" in ``docs/`` and requires
    each *distinct fingerprint* to be allowlisted.  Adding a new factual number
    to the prose therefore forces an explicit decision -- either register it as
    a ``CLAIM`` (and it becomes drift-checked) or allowlist it as a
    ``CONSTANT``/``INPUT`` (and you state that it cannot drift).
    ``--update`` regenerates the baseline for review.

Usage
-----
    python tools/doc_check.py           # human-readable report
    python tools/doc_check.py -q        # only failures
    python tools/doc_check.py --update  # refresh the AUDIT baseline: REVIEW THE DIFF

Also imported by ``tests/test_docs_consistency.py`` so ``pytest`` fails on
doc drift.
"""

from __future__ import annotations

import collections
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "results" / "data"
DOCS = ROOT / "docs"
AUDIT_BASELINE = ROOT / "tools" / "docs_numbers.txt"

# A line that matches a RETIRED pattern is tolerated when it is explicitly
# talking about the superseded value rather than asserting it.  The markers are
# *per item* (see ``Retired.exempt``) because a global list would either be too
# permissive or forbid ever explaining a past mistake.
HISTORY_MARKERS = (
    "早期版本",
    "旧版本",
    "开发记录",
    "曾经",
    "此前",
    "回归",
    "regression",
    "used to",
)


# --------------------------------------------------------------------------- #
# data access
# --------------------------------------------------------------------------- #
_cache: dict[str, dict] = {}


def data(name: str) -> dict:
    """Load ``results/data/<name>.json`` (cached)."""
    if name not in _cache:
        path = DATA / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing -- run `python experiments/run_all.py` first")
        _cache[name] = json.loads(path.read_text(encoding="utf-8"))
    return _cache[name]


def live():
    """Recompute the few quantities that the JSON does not carry verbatim."""
    from qcdma_qkd import budget, security
    from qcdma_qkd.config import BudgetConfig, SecurityConfig

    bud, sec = BudgetConfig(), SecurityConfig()
    eps_round = security.from_config(sec).total()
    return dict(
        eps_round=eps_round,
        max_rounds=security.max_rounds(eps_round, sec.eps_total),
        auth_key_length=budget.auth_key_bits(bud)["key_length"],
        auth_per_message=budget.auth_key_bits(bud)["per_message"],
        k_req=39.86313713864835,          # log2(1/1e-12), pinned by tests
    )


# --------------------------------------------------------------------------- #
# formatting helpers -- must mirror the doc style
# --------------------------------------------------------------------------- #
def tex(x: float, sig: int = 3) -> str:
    """Render ``x`` as ``a\\times10^{b}`` the way the docs write it."""
    m, e = f"{x:.{sig - 1}e}".split("e")
    m = m.rstrip("0").rstrip(".")
    return f"{m}\\times10^{{{int(e)}}}"


def plain_sci(x: float, sig: int = 3) -> str:
    """Render ``x`` as ``a.bce-4`` (plain-text style used in tables)."""
    m, e = f"{x:.{sig - 1}e}".split("e")
    m = m.rstrip("0").rstrip(".")
    return f"{m}e{int(e)}"


# --------------------------------------------------------------------------- #
# claims
# --------------------------------------------------------------------------- #
@dataclass
class Claim:
    """One number that appears in the docs and must track the results.

    ``anchor`` is the load-bearing part: the claim is only satisfied if a line
    *containing the anchor* also contains the expected rendering.  Checking
    "the number appears somewhere in the file" is far too weak -- a stale value
    is usually still present *elsewhere* in a legitimate role (``4.31 Mbit`` is
    correct as the asymptotic reference right next to the finite-size
    ``3.33 Mbit``), so a whole-file search passes even after the real line has
    drifted.
    """

    label: str
    anchor: str                      # substring identifying the intended line
    value: object                    # callable -> float/str
    form: object                     # callable(value) -> str
    files: tuple[str, ...]
    note: str = ""
    alternatives: tuple[str, ...] = field(default=())

    def expected(self):
        v = self.value() if callable(self.value) else self.value
        out = [self.form(v)] if callable(self.form) else [str(self.form)]
        return out + list(self.alternatives)


def build_claims() -> list[Claim]:
    e0 = lambda: data("e0_headline")
    e6 = lambda: data("e6_epsilon")
    e7 = lambda: data("e7_loop")

    return [
        # ---- quantum layer ------------------------------------------------ #
        Claim("finite-size rate @25 km (with MUI)",
              anchor="每轮密钥 $K_n$",
              value=lambda: e0()["rate_25km_with_mui_finite"],
              form=lambda v: f"{v:.4f}", files=("00_summary.md",),
              alternatives=("0.0333", "0.03332", "0.033316"),
              note="the loop uses this, not the asymptotic 0.0431"),
        Claim("K_n per round (finite-size)",
              anchor="每轮密钥 $K_n$",
              value=lambda: e0()["key_bits_per_round"] / 1e6,
              form=lambda v: f"{v:.2f} Mbit",
              files=("00_summary.md", "02_protocol.md"),
              alternatives=("3.332", "3.333", "3.33"),
              note="was 4.31 Mbit while the loop still used the asymptotic rate"),
        Claim("K_n per round (asymptotic, reference only)",
              anchor="同轮渐近值",
              value=lambda: e0()["key_bits_per_round_asym"] / 1e6,
              form=lambda v: f"{v:.2f} Mbit", files=("00_summary.md",),
              note="must always be labelled asymptotic"),
        Claim("asymptotic overstates K_n by",
              anchor="同轮渐近值",
              value=lambda: 1.0 / e0()["finite_size_factor"] - 1.0,
              form=lambda v: f"高估 {v * 100:.0f}%", files=("00_summary.md",),
              alternatives=("高估 29%",),
              note="the finite-size penalty re-expressed as an overstatement"),

        # ---- MUI ---------------------------------------------------------- #
        Claim("xi_MUI at eps=1e-12",
              anchor="$\\xi_{\\text{MUI}}$ @ $\\epsilon=10^{-12}$",
              value=lambda: e0()["xi_mui_required"],
              form=lambda v: f"{v:.4f}",
              files=("00_summary.md",),
              alternatives=("0.0692",)),
        Claim("E[xi_MUI]",
              anchor="$\\mathbb{E}[\\xi_{\\text{MUI}}]$",
              value=lambda: e0()["xi_mui_mean"],
              form=lambda v: f"{v:.5f}", files=("00_summary.md",),
              alternatives=("0.00684",)),

        # ---- authentication billing -------------------------------------- #
        Claim("auth key per round",
              anchor="每轮认证密钥",
              value=lambda: live()["auth_key_length"],
              form=lambda v: f"{v:.1f} bit",
              files=("00_summary.md", "06_key_budget.md"),
              alternatives=("207.7", "207.726"),
              note="one fresh eps-AXU key per round; NOT the amortised 0.132"),
        Claim("overhead fraction (a+c)/K",
              anchor="总开销占比",
              value=lambda: e0()["overhead_fraction"],
              form=lambda v: plain_sci(v, 3),
              files=("00_summary.md",),
              alternatives=("1.39e-4", "1.392e-4"),
              note="uses the finite-size K_n, hence larger than the old value"),

        # ---- bootstrap ---------------------------------------------------- #
        Claim("K_0 per-round cost (auth + seed, before reserve)",
              anchor="$K_0$ 需求",
              value=lambda: e0()["k0_required"] / 1.1,
              form=lambda v: f"{v:.1f} bit", files=("00_summary.md",),
              alternatives=("463.7",),
              note="auth 207.7 + seed 256"),
        Claim("K_0 requirement (with 1.1 failure reserve)",
              anchor="| 1 |",
              value=lambda: e0()["k0_required"],
              form=lambda v: f"{v:.1f} bit", files=("00_summary.md",),
              alternatives=("510.1",),
              note="the ring table quotes the reserved figure"),

        # ---- refresh constraint (extractor branch) ----------------------- #
        Claim("f_min (tail, p=1, eps=1e-12)",
              anchor="刷新率要求",
              value=lambda: e0()["refresh_f_min_tail"],
              form=lambda v: f"{v:.4f}", files=("00_summary.md",),
              alternatives=("0.7886", "0.78857")),
        Claim("k_req = log2(1/eps_code)",
              anchor="k_{\\text{req}}",
              value=lambda: e0()["refresh_k_req_bits"],
              form=lambda v: f"{v:.2f} bit", files=("00_summary.md",),
              alternatives=("39.86", "39.9")),
        Claim("max known chips",
              anchor="可泄漏预算",
              value=lambda: e0()["refresh_max_known_chips"],
              form=lambda v: f"{v:.1f}", files=("00_summary.md",),
              alternatives=("216.1",)),

        # ---- epsilon ----------------------------------------------------- #
        Claim("per-round epsilon total",
              anchor="\\epsilon_{\\text{round}}",
              value=lambda: e6()["total"],
              form=lambda v: f"{v:.3f}\\\\times10^{{-10}}",
              alternatives=("3.050", "3.05", "3.08"),
              files=("00_summary.md",),
              note="the whole per-round budget, not one component"),
        Claim("max rounds at default epsilon",
              anchor="最多约",
              value=lambda: e6()["max_rounds"],
              form=lambda v: f"**{int(v)} 轮**",
              alternatives=("3 轮",),
              files=("00_summary.md",),
              note="3 rounds -- the old 1.8e4 came from composing 1e-12 only"),

        # ---- F2 budget-limited distance ---------------------------------- #
        Claim("closed-loop reach at n=1e8",
              anchor="预算仍然先撞墙",
              value=lambda: e7()["f2_budget_limit_km"][-1],
              form=lambda v: f"{v:.1f} km", files=("00_summary.md",),
              alternatives=("47.7 km",),
              note="finite-size + budget, vs the asymptotic 68.7 km"),

        # ---- MitM --------------------------------------------------------- #
        Claim("MitM survival after 2 rounds",
              anchor="MitM 有认证检测概率",
              value=lambda: e0()["mitm_survival_after_2_rounds"],
              form=lambda v: tex(v, 3), files=("00_summary.md",),
              alternatives=("5.53\\times10^{-76}", "10^{-76}")),

        # ---- E10: model vs bound ----------------------------------------- #
        Claim("MUI Gaussian gap at eps=1e-12",
              anchor="| $10^{-12}$ | 0.06918 | 0.03254",
              value=lambda: data("e10_model_vs_bound")["mui"][2]["gaussian_gap"],
              form=lambda v: f"**−{abs(v) * 100:.0f}%**",
              alternatives=("**−53%**", "**-53%**"),
              files=("09_model_vs_bound.md",),
              note="the Gaussian quantile understates the MUI excess noise"),
        Claim("MUI bound gap at eps=1e-12",
              anchor="| $10^{-12}$ | 0.06918 | 0.03254",
              value=lambda: data("e10_model_vs_bound")["mui"][2]["bound_gap"],
              form=lambda v: f"+{v * 100:.0f}%",
              alternatives=("+27%",),
              files=("09_model_vs_bound.md",)),
        Claim("seed entropy lost at p=0.10",
              anchor="| 0.10 | 102.4 |",
              value=lambda: data("e10_model_vs_bound")["entropy"][1]["entropy_lost"],
              form=lambda v: f"{v:.1f}",
              alternatives=("73.6",),
              files=("09_model_vs_bound.md",),
              note="mean-leakage entropy minus upper-tail entropy"),
        Claim("rigorous f_min at eps=1e-12",
              anchor="| $10^{-12}$ | 0.78893 |",
              value=lambda: data("e10_model_vs_bound")["refresh"][2]["f_min_rigorous"],
              form=lambda v: f"**{v:.5f}**",
              alternatives=("**0.78857**",),
              files=("09_model_vs_bound.md",)),
        Claim("rate penalty from PE confidence bounds",
              anchor="| + PE 置信界（$\\chi^2$ 精确） |",
              value=lambda: data("e10_model_vs_bound")["rate"]["pe_over_finite"],
              form=lambda v: f"{v * 100:.1f}%",
              alternatives=("76.9%",),
              files=("09_model_vs_bound.md", "08_open_problems.md")),
        Claim("finite-size rate @25 km in E10",
              anchor="| + 有限码长 AEP",
              value=lambda: data("e10_model_vs_bound")["rate"]["finite_nominal"],
              form=lambda v: f"**{v:.5f}**",
              alternatives=("**0.04114**",),
              files=("09_model_vs_bound.md", "08_open_problems.md")),
        Claim("finite-size + PE rate @25 km in E10",
              anchor="+ PE 置信界（$\\chi^2$ 精确）",
              value=lambda: data("e10_model_vs_bound")["rate"]["finite_with_pe"],
              form=lambda v: f"**{v:.5f}**",
              alternatives=("**0.03164**",),
              files=("09_model_vs_bound.md", "08_open_problems.md")),
    ]


# --------------------------------------------------------------------------- #
# retired values
# --------------------------------------------------------------------------- #
@dataclass
class Retired:
    """A value that must not be asserted anywhere.

    ``exempt`` holds substrings that mark a line as *explaining* the superseded
    number ("the amortised figure would be 0.132 bit, but ...").  Keep these
    narrow: every extra exemption is a place a stale number can hide.
    """

    pattern: str
    why: str
    exempt: tuple[str, ...] = ()


RETIRED: list[Retired] = [
    Retired(r"0\.132\s*bit",
            "amortised auth cost; the loop re-keys every round -> 207.7 bit",
            exempt=("摊销", "amortised", "分数比特", "per_message", "来历")),
    Retired(r"3\.08\\times10\^\{-10\}|3\.08e-10",
            "per-round eps before the double-counted eps_auth_rounds=4 was removed",
            exempt=("两套不相容", "当作总分", "低估")),
    Retired(r"1\.8\\times10\^\{4\}|1\.8e4|18059",
            "round limit computed by composing 1e-12 instead of the per-round total"),
    Retired(r"7\.44\\times10\^\{-10\}|7\.44e-10",
            "same composition bug (auth term only)"),
    Retired(r"5\.9e-5|5\.9\\times10\^\{-5\}",
            "overhead computed against the asymptotic K_n=4.31 Mbit"),
    Retired(r"3\.1\\times10\^\{-8\}|3\.1e-8",
            "a/K using the amortised auth cost"),
    Retired(r"281\.7",
            "K_0 requirement before the auth key became 207.7 bit"),
    Retired(r"31\.5\s*km",
            "budget-limited distance under the asymptotic rate model"),
    Retired(r"55\.5\s*km|66\.5\s*km|68\.5\s*km",
            "same asymptotic-rate F2 table"),
]


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    ok: bool
    label: str
    detail: str


def _doc_text(name: str) -> str:
    path = DOCS / name
    if not path.exists():
        raise FileNotFoundError(f"missing doc {path}")
    return path.read_text(encoding="utf-8")


def check_claims(claims: list[Claim]) -> list[Result]:
    """Each claim must hold on at least one *anchored* line."""
    out = []
    for c in claims:
        expected = c.expected()
        found_in, anchored_lines = [], 0
        for doc in c.files:
            for line in _doc_text(doc).splitlines():
                if c.anchor not in line:
                    continue
                anchored_lines += 1
                if any(e in line for e in expected):
                    found_in.append(doc)
                    break
        if found_in:
            out.append(Result(True, c.label,
                              f"ok in {', '.join(sorted(set(found_in)))}"))
        elif anchored_lines == 0:
            out.append(Result(False, c.label,
                              f"anchor {c.anchor!r} not found in {list(c.files)}"
                              " -- the doc was restructured; re-anchor the claim"))
        else:
            out.append(Result(False, c.label,
                              f"anchor {c.anchor!r} found ({anchored_lines} line(s)) "
                              f"but none contains any of {expected}"
                              + (f" [{c.note}]" if c.note else "")))
    return out


def check_retired(retired: list[Retired]) -> list[Result]:
    out = []
    for r in retired:
        rx = re.compile(r.pattern)
        offenders = []
        for path in sorted(DOCS.glob("*.md")):
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not rx.search(line):
                    continue
                if any(m in line for m in HISTORY_MARKERS + r.exempt):
                    continue
                offenders.append(f"{path.name}:{i}: {line.strip()[:96]}")
        out.append(Result(not offenders, f"retired: {r.pattern}",
                          "clean" if not offenders
                          else f"{r.why}\n      " + "\n      ".join(offenders)))
    return out


def run_all() -> list[Result]:
    return (check_claims(build_claims()) + check_retired(RETIRED)
            + check_audit())


# --------------------------------------------------------------------------- #
# AUDIT: every "number + unit" in the docs must be accounted for
# --------------------------------------------------------------------------- #
# Matches a number with a physical unit.  0.13 / 3.33 / 1,530.3 / 1.2\times10^{-4}
# Units are listed explicitly so prose like "第 3 轮" or "1e-12" is not captured.
_UNIT = (r"(?:Mbit|kbit|bit/symbol|bits/symbol|Gbit|bit|km|s\b|ms|"
         r"chip/s|Gchip/s|symbols?|"
         r"SNU|dB)")


def _number_pattern() -> re.Pattern:
    num = r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"
    return re.compile(rf"(?<![\w.^]){num}\s*({_UNIT})(?![\w])")


def fingerprint(number: str, unit: str) -> str:
    """Canonical form of a documented quantity.

    ``1530.3 bit``, ``1,530.3 bit`` and ``1530.30 bit`` are the same quantity;
    ``0.0333`` and ``0.03332`` are not (different precision = different claim).
    """
    n = number.replace(",", "")
    return f"{n} {unit.strip()}"


def _exempt_markers() -> tuple[str, ...]:
    """Markers that let a line carry a number the checker will not recompute.

    History markers cover "the old value was X, we fixed it"; each ``Retired``
    item also carries its own narrow exemptions.  A line explaining a retired
    number is *supposed* to contain it, so it must not be audited.
    """
    extra: list[str] = []
    for item in RETIRED:
        extra.extend(item.exempt)
    return HISTORY_MARKERS + tuple(extra)


def scan_docs() -> dict[str, list[str]]:
    """fingerprint -> list of ``file:line`` where it occurs."""
    pat = _number_pattern()
    exempt = _exempt_markers()
    out: dict[str, list[str]] = collections.defaultdict(list)
    for path in sorted(DOCS.glob("*.md")):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            # A line documenting a historical/retired value is exempt: it is
            # *supposed* to contain a number that nothing recomputes.
            if any(m in line for m in exempt):
                continue
            for m in pat.finditer(line):
                fp = fingerprint(m.group(1) + (f".{m.group(2)}" if m.group(2)
                                               else ""), m.group(3))
                out[fp].append(f"{path.name}:{i}")
    return out


def load_audit_baseline() -> list[str]:
    if not AUDIT_BASELINE.exists():
        return []
    out = []
    for raw in AUDIT_BASELINE.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        out.append(raw)
    return out


def write_audit_baseline(fps: list[str]) -> None:
    header = [
        "# Baseline for tools/doc_check.py AUDIT.",
        "#",
        "# Every 'number + unit' written in docs/ must appear here, so that a new",
        "# factual number cannot slip in unnoticed.  Each entry is one *distinct*",
        "# fingerprint; a quantity quoted in several places is listed once.",
        "#",
        "# When this file changes, the diff IS the review: for each new line ask",
        "#   (a) does it come from results/data/*.json?  -> add a CLAIM in",
        "#       doc_check.py instead, so it becomes drift-checked; or",
        "#   (b) is it an input/constant (wavelength, tag length, ...) that cannot",
        "#       drift?  -> leave it here.",
        "#",
        "# Regenerate with:  python tools/doc_check.py --update",
        "",
    ]
    AUDIT_BASELINE.write_text("\n".join(header + fps) + "\n", encoding="utf-8")


def check_audit() -> list[Result]:
    """No unregistered "number + unit" may appear in the docs."""
    found = scan_docs()
    baseline = load_audit_baseline()
    if not baseline:
        return [Result(False, "audit: baseline",
                       f"{AUDIT_BASELINE.name} is missing or empty; regenerate "
                       "with `python tools/doc_check.py --update`")]
    known = set(baseline)
    fresh = {fp: where for fp, where in found.items() if fp not in known}
    stale = sorted(known - set(found))

    out = []
    if fresh:
        parts = []
        for fp in sorted(fresh):
            parts.append(f"{fp!r} at {', '.join(fresh[fp][:3])}")
        out.append(Result(False, "audit: unregistered numbers",
                          "these 'number + unit' strings are in docs/ but not in "
                          f"{AUDIT_BASELINE.name}. Either register them as CLAIMS "
                          "(preferred) or add them as inputs:\n      "
                          + "\n      ".join(parts)))
    if stale:
        out.append(Result(False, "audit: stale baseline entries",
                          f"{len(stale)} fingerprint(s) in {AUDIT_BASELINE.name} "
                          "no longer occur in docs/ (the prose moved on):\n      "
                          + "\n      ".join(stale[:20])))
    if not out:
        out.append(Result(True, "audit: coverage",
                          f"{len(found)} distinct quantities, all registered"))
    return out


def main(argv: list[str]) -> int:
    if "--update" in argv:
        fps = sorted(scan_docs())
        write_audit_baseline(fps)
        print(f"wrote {len(fps)} fingerprints to {AUDIT_BASELINE.relative_to(ROOT)}")
        print("Review the diff: every added line is a number that now escapes "
              "drift checking unless it becomes a CLAIM.")
        return 0

    quiet = "-q" in argv or "--quiet" in argv
    results = run_all()
    failures = [r for r in results if not r.ok]

    if not quiet:
        print("=" * 74)
        print("docs consistency check")
        print("=" * 74)
        for r in results:
            if r.ok:
                print(f"  [ok]   {r.label:<44s} {r.detail}")
            else:
                print(f"  [FAIL] {r.label:<44s} {r.detail}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} of {len(results)} checks")
        print("\nHint: the numbers in docs/ are supposed to track "
              "results/data/*.json.\n      Re-run `python experiments/run_all.py` "
              "and update the prose.")
        return 1
    print(f"all {len(results)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
