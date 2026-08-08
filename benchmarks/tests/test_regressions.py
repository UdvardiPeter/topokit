# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Unit tests for the perf-regression comparator (no benchmark run needed)."""

from typing import Any

from topokit_bench.regressions import check_regressions


def _report(cases: list[dict[str, Any]], **meta: Any) -> dict[str, Any]:
    base_meta = {"system": "Linux", "machine": "x86_64", "pyamg": "5.3.0"}
    base_meta.update(meta)
    return {"meta": base_meta, "cases": cases}


def _case(**over: Any) -> dict[str, Any]:
    case = {
        "label": "cantilever_3d_20",
        "wall_per_iter_s": 1.0,
        "peak_rss_kb": 1000,
        "amg_iterations": 20,
    }
    case.update(over)
    return case


def test_identical_reports_pass() -> None:
    failures, notes = check_regressions(_report([_case()]), _report([_case()]))
    assert failures == []
    assert notes == []


def test_wall_regression_beyond_tolerance_fails() -> None:
    failures, _ = check_regressions(_report([_case()]), _report([_case(wall_per_iter_s=1.31)]))
    assert len(failures) == 1
    assert "wall_per_iter_s" in failures[0]


def test_wall_within_tolerance_passes() -> None:
    failures, _ = check_regressions(_report([_case()]), _report([_case(wall_per_iter_s=1.29)]))
    assert failures == []


def test_rss_regression_fails_at_ten_percent() -> None:
    failures, _ = check_regressions(_report([_case()]), _report([_case(peak_rss_kb=1101)]))
    assert len(failures) == 1
    assert "peak_rss_kb" in failures[0]


def test_improvements_never_fail() -> None:
    better = _case(wall_per_iter_s=0.1, peak_rss_kb=10, amg_iterations=1)
    failures, _ = check_regressions(_report([_case()]), _report([better]))
    assert failures == []


def test_amg_absolute_slack_allows_small_counts() -> None:
    # +2 on a count of 11 is 18%, over the 10% band but within the absolute slack
    failures, _ = check_regressions(
        _report([_case(amg_iterations=11)]), _report([_case(amg_iterations=13)])
    )
    assert failures == []
    failures, _ = check_regressions(
        _report([_case(amg_iterations=11)]), _report([_case(amg_iterations=14)])
    )
    assert len(failures) == 1
    assert "amg_iterations" in failures[0]


def test_platform_mismatch_fails_and_stops() -> None:
    failures, _ = check_regressions(
        _report([_case()], system="Darwin"), _report([_case(wall_per_iter_s=99.0)])
    )
    assert len(failures) == 1  # platform only; per-case comparison is not attempted
    assert "system" in failures[0]


def test_machine_mismatch_fails() -> None:
    failures, _ = check_regressions(_report([_case()], machine="arm64"), _report([_case()]))
    assert any("machine" in f for f in failures)


def test_pyamg_change_disables_only_the_amg_gate() -> None:
    failures, notes = check_regressions(
        _report([_case()], pyamg="5.2.0"), _report([_case(amg_iterations=99)])
    )
    assert failures == []
    assert any("pyamg" in n for n in notes)


def test_errored_case_fails() -> None:
    failures, _ = check_regressions(
        _report([_case()]), _report([{"label": "cantilever_3d_20", "error": "boom"}])
    )
    assert len(failures) == 1
    assert "failed to run" in failures[0]


def test_case_missing_from_run_is_a_note() -> None:
    failures, notes = check_regressions(
        _report([_case(), _case(label="cantilever_3d_60")]), _report([_case()])
    )
    assert failures == []
    assert any("cantilever_3d_60" in n for n in notes)


def test_case_missing_from_baseline_is_a_note() -> None:
    failures, notes = check_regressions(
        _report([_case()]), _report([_case(), _case(label="new_case")])
    )
    assert failures == []
    assert any("new_case" in n for n in notes)


def test_missing_metric_is_a_note_not_a_failure() -> None:
    failures, notes = check_regressions(
        _report([_case(amg_iterations=None)]), _report([_case(amg_iterations=None)])
    )
    assert failures == []
    assert notes == []
