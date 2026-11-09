"""Lock the documentation to the computed results.

The project has twice shipped docs whose numbers were correct when typed and
wrong afterwards, with **every other test still green**: code and prose drift
independently, and unit tests cannot see prose.  These tests import
``tools/doc_check.py`` and fail on that drift.

Running ``python experiments/run_all.py`` refreshes ``results/data/*.json``;
these tests then assert that the digits quoted in ``docs/`` still match.

If a test here fails, the fix is *not* to loosen the assertion -- it is to look
at which of the two is wrong:

* results changed and the doc is stale  -> update the doc;
* the doc is right and the results moved for a bad reason -> fix the code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import doc_check  # noqa: E402


def test_results_are_present():
    """The check is meaningless on a stale/absent results directory."""
    assert doc_check.DATA.is_dir()
    missing = [n for n in ("e0_headline", "e6_epsilon", "e7_loop")
               if not (doc_check.DATA / f"{n}.json").exists()]
    assert not missing, (
        f"missing {missing}; run `python experiments/run_all.py` first")


@pytest.mark.parametrize("claim", doc_check.build_claims(),
                         ids=lambda c: c.label)
def test_doc_quotes_the_computed_value(claim):
    """Every documented number must still be derivable from the JSON.

    ``Claim.expected()`` recomputes from ``results/data`` (never from a
    hand-typed constant), so this catches the failure mode where results move
    and the prose does not.
    """
    res = doc_check.check_claims([claim])[0]
    assert res.ok, res.detail


@pytest.mark.parametrize("item", doc_check.RETIRED,
                         ids=lambda r: r.pattern)
def test_retired_values_do_not_reappear(item):
    """Superseded numbers must not be re-asserted.

    This is the check that would have caught all three of the stale-value bugs
    (0.132 bit auth cost, 1.8e4 round limit, 31.5 km budget reach) the moment
    the underlying model changed.  Lines that explicitly discuss the history are
    exempt via ``Retired.exempt``.
    """
    res = doc_check.check_retired([item])[0]
    assert res.ok, res.detail


def test_every_doc_number_render_is_unique_enough():
    """Guard the guard: a claim whose expected string is trivially common.

    A one-character expectation (``"1"``) would pass trivially.  Requiring some
    specificity keeps ``CLAIMS`` honest as it grows.
    """
    for claim in doc_check.build_claims():
        for text in claim.expected():
            assert len(text) >= 3, (
                f"claim {claim.label!r} expects the very short string {text!r}; "
                "it would match almost anything")
