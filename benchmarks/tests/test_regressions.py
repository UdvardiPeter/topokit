# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Unit tests for the perf-regression comparator (no benchmark run needed)."""

from typing import Any

import pytest

from topokit_bench.regressions import check_regressions


def _report(cases: list[dict[str, Any]], **meta: Any) -> dict[str, Any]:
    return {
        "meta": {"system": "Linux", "machine": "x86_64", "pyamg": "5.3.0", **meta},
        "cases": cases,
    }


def _case(**over: Any) -> dict[str, Any]:
    return {
        "label": "cantilever_3d_20",
        "elements": 8000,
        "dof": 26460,
        "solver": "AmgCG",
        "wall_per_iter_s": 1.0,
        "peak_rss_kb": 1000,
        "amg_iterations": 20,
    } | over


def _check(base: list[dict[str, Any]], run: list[dict[str, Any]], **meta: Any) -> list[str]:
    return check_regressions(_report(base, **meta), _report(run))


@pytest.mark.parametrize(
    "run,expected",
    [
        (_case(), None),
        (_case(wall_per_iter_s=1.29), None),  # inside the 30% wall band
        (_case(wall_per_iter_s=1.31), "wall_per_iter_s"),
        (_case(peak_rss_kb=1101), "peak_rss_kb"),  # the 10% RSS band
        (_case(wall_per_iter_s=0.1, peak_rss_kb=10, amg_iterations=1), None),  # improvements
        (_case(peak_rss_kb=None), "peak_rss_kb present in the baseline"),
        # a solver falling back to a direct factorization drops the AMG count
        (_case(amg_iterations=None), "amg_iterations present in the baseline"),
        (_case(wall_per_iter_s=0.0), "wall_per_iter_s present in the baseline"),
    ],
)
def test_metric_gates(run: dict[str, Any], expected: str | None) -> None:
    failures = _check([_case()], [run])
    if expected is None:
        assert failures == []
    else:
        assert len(failures) == 1
        assert expected in failures[0]


@pytest.mark.parametrize("count,fails", [(13, False), (14, True)])
def test_amg_absolute_slack_allows_small_counts(count: int, fails: bool) -> None:
    # +2 on a count of 11 is 18%, over the 10% band but within the absolute slack
    failures = _check([_case(amg_iterations=11)], [_case(amg_iterations=count)])
    assert bool(failures) is fails


@pytest.mark.parametrize("field,value", [("elements", 1000), ("dof", 999), ("solver", "Direct")])
def test_a_redefined_case_fails_and_counts_toward_nothing(field: str, value: Any) -> None:
    # shrink a case in CASES and every metric "improves" against a bigger problem;
    # pyamg missing and auto_solver falling back to Direct reads the same way
    failures = _check([_case()], [_case(wall_per_iter_s=99.0, **{field: value})])
    assert any(f"{field} " in f and "must be regenerated" in f for f in failures)
    assert any("no metric was compared" in f for f in failures)


@pytest.mark.parametrize("meta", [{"system": "Darwin"}, {"machine": "arm64"}, {"system": None}])
def test_a_platform_mismatch_fails_before_any_case(meta: dict[str, Any]) -> None:
    failures = _check([_case()], [_case(wall_per_iter_s=99.0)], **meta)
    assert len(failures) == 1  # platform only; per-case comparison is not attempted
    assert "regenerate the baseline on this platform" in failures[0]


def test_reports_without_meta_do_not_pass_on_none_equals_none() -> None:
    failures = check_regressions({"cases": [_case()]}, {"cases": [_case()]})
    assert len(failures) == 2  # one per platform key
    assert any("system" in f for f in failures)
    assert any("machine" in f for f in failures)


def test_a_pyamg_change_disables_only_the_amg_gate() -> None:
    assert _check([_case()], [_case(amg_iterations=99)], pyamg="5.2.0") == []
    failures = _check([_case()], [_case(peak_rss_kb=1101)], pyamg="5.2.0")
    assert len(failures) == 1
    assert "peak_rss_kb" in failures[0]


def test_an_errored_case_fails_and_gates_nothing() -> None:
    failures = _check([_case()], [{"label": "cantilever_3d_20", "error": "boom"}])
    assert any("failed to run" in f for f in failures)
    assert any("no metric was compared" in f for f in failures)


def test_an_empty_baseline_gates_nothing_and_fails() -> None:
    assert _check([], [_case()]) == [
        "no metric was compared against the baseline; the run is not gated"
    ]


def test_near_threshold_failures_do_not_round_into_the_limit() -> None:
    # a +10.0% delta against a +10% limit must not render as "+10% vs +10%"
    failures = _check([_case()], [_case(peak_rss_kb=1101)])
    assert "1101 vs baseline 1000 (+10.1%, limit 1100)" in failures[0]
    failures = _check([_case(amg_iterations=25)], [_case(amg_iterations=28)])
    assert "limit 27.5" in failures[0]
