# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Unit tests for the perf-regression comparator (no benchmark run needed)."""

from typing import Any

from topokit_bench.regressions import check_regressions


def _report(cases: list[dict[str, Any]], **meta: Any) -> dict[str, Any]:
    base_meta = {"system": "Linux", "machine": "x86_64", "pyamg": "5.3.0"}
    base_meta.update(meta)
    return {"meta": base_meta, "cases": cases}


def _case(**over: Any) -> dict[str, Any]:
    case: dict[str, Any] = {
        "label": "cantilever_3d_20",
        "elements": 8000,
        "dof": 26460,
        "solver": "AmgCG",
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


def test_element_count_change_fails_the_case_and_skips_its_metrics() -> None:
    # shrink a case in CASES and every metric "improves" against a bigger problem
    failures, _ = check_regressions(
        _report([_case(), _case(label="mbb_150x50_full")]),
        _report([_case(elements=1000, wall_per_iter_s=99.0), _case(label="mbb_150x50_full")]),
    )
    assert len(failures) == 1
    assert "elements 8000 in the baseline but 1000 in this run" in failures[0]
    assert "the baseline must be regenerated" in failures[0]


def test_dof_change_fails_the_case() -> None:
    failures, _ = check_regressions(
        _report([_case(), _case(label="mbb_150x50_full")]),
        _report([_case(dof=999), _case(label="mbb_150x50_full")]),
    )
    assert len(failures) == 1
    assert "dof 26460 in the baseline but 999 in this run" in failures[0]


def test_solver_change_fails_the_case() -> None:
    # pyamg missing, auto_solver falls back to Direct: same class of silent switch
    failures, _ = check_regressions(
        _report([_case(), _case(label="mbb_150x50_full")]),
        _report([_case(solver="Direct"), _case(label="mbb_150x50_full")]),
    )
    assert len(failures) == 1
    assert "solver 'AmgCG' in the baseline but 'Direct' in this run" in failures[0]


def test_a_changed_case_does_not_count_toward_the_gated_total() -> None:
    failures, _ = check_regressions(_report([_case()]), _report([_case(solver="Direct")]))
    assert len(failures) == 2
    assert any("solver" in f for f in failures)
    assert any("no metric was compared" in f for f in failures)


def test_identity_field_absent_from_both_sides_is_not_a_mismatch() -> None:
    bare: dict[str, Any] = {"label": "cantilever_3d_20", "wall_per_iter_s": 1.0}
    failures, notes = check_regressions(_report([dict(bare)]), _report([dict(bare)]))
    assert failures == []
    assert not any("regenerated" in n for n in notes)


def test_platform_mismatch_fails_and_stops() -> None:
    failures, _ = check_regressions(
        _report([_case()], system="Darwin"), _report([_case(wall_per_iter_s=99.0)])
    )
    assert len(failures) == 1  # platform only; per-case comparison is not attempted
    assert "system" in failures[0]


def test_machine_mismatch_fails() -> None:
    failures, _ = check_regressions(_report([_case()], machine="arm64"), _report([_case()]))
    assert any("machine" in f for f in failures)


def test_missing_meta_key_fails() -> None:
    failures, _ = check_regressions(_report([_case()], system=None), _report([_case()]))
    assert len(failures) == 1
    assert "system missing" in failures[0]
    assert "regenerate the baseline" in failures[0]


def test_reports_without_meta_do_not_pass_on_none_equals_none() -> None:
    failures, _ = check_regressions({"cases": [_case()]}, {"cases": [_case()]})
    assert len(failures) == 2  # one per platform key, and no case comparison
    assert any("system" in f for f in failures)
    assert any("machine" in f for f in failures)


def test_unlabelled_run_case_fails() -> None:
    failures, _ = check_regressions(
        _report([_case()]), _report([_case(), {"wall_per_iter_s": 1.0}])
    )
    assert len(failures) == 1
    assert "no label" in failures[0]


def test_duplicate_label_in_the_run_fails() -> None:
    failures, _ = check_regressions(_report([_case()]), _report([_case(), _case()]))
    assert len(failures) == 1
    assert "appears 2 times in the run" in failures[0]


def test_duplicate_label_in_the_baseline_fails() -> None:
    failures, _ = check_regressions(_report([_case(), _case()]), _report([_case()]))
    assert len(failures) == 1
    assert "appears 2 times in the baseline" in failures[0]


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
    # the only case errored, so the nothing-gated guard fires alongside it
    assert len(failures) == 2
    assert any("failed to run" in f for f in failures)
    assert any("no metric was compared" in f for f in failures)


def test_errored_baseline_case_is_a_note() -> None:
    failures, notes = check_regressions(
        _report([{"label": "cantilever_3d_20", "error": "boom"}, _case(label="mbb_150x50_full")]),
        _report([_case(), _case(label="mbb_150x50_full")]),
    )
    assert failures == []
    assert any("cantilever_3d_20: no baseline entry" in n for n in notes)


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


def test_amg_absent_from_both_sides_is_silent() -> None:
    # 2D direct-solver cases have no AMG count on either side; not even a note
    failures, notes = check_regressions(
        _report([_case(amg_iterations=None)]), _report([_case(amg_iterations=None)])
    )
    assert failures == []
    assert notes == []


def test_metric_absent_from_both_sides_is_a_note() -> None:
    failures, notes = check_regressions(
        _report([_case(peak_rss_kb=None)]), _report([_case(peak_rss_kb=None)])
    )
    assert failures == []
    assert any("peak_rss_kb absent from both" in n for n in notes)


def test_metric_only_in_the_run_is_a_note() -> None:
    failures, notes = check_regressions(_report([_case(peak_rss_kb=None)]), _report([_case()]))
    assert failures == []
    assert any("peak_rss_kb not in the baseline" in n for n in notes)


def test_metric_dropped_by_the_run_fails() -> None:
    failures, _ = check_regressions(_report([_case()]), _report([_case(peak_rss_kb=None)]))
    assert len(failures) == 1
    assert "peak_rss_kb present in the baseline" in failures[0]
    # a solver falling back to a direct factorization drops the AMG count
    failures, _ = check_regressions(_report([_case()]), _report([_case(amg_iterations=None)]))
    assert len(failures) == 1
    assert "amg_iterations present in the baseline" in failures[0]


def test_non_positive_run_value_fails() -> None:
    failures, _ = check_regressions(_report([_case()]), _report([_case(wall_per_iter_s=0.0)]))
    assert len(failures) == 1
    assert "implausible" in failures[0]


def test_non_positive_baseline_value_is_a_note() -> None:
    failures, notes = check_regressions(_report([_case(peak_rss_kb=0)]), _report([_case()]))
    assert failures == []
    assert any("not usable" in n for n in notes)


def test_empty_baseline_gates_nothing_and_fails() -> None:
    failures, notes = check_regressions(_report([]), _report([_case()]))
    assert len(failures) == 1
    assert "no metric was compared" in failures[0]
    assert any("no baseline entry" in n for n in notes)


def test_near_threshold_failures_report_one_decimal() -> None:
    # a +10.0% delta against a +10% limit must not render as "+10% vs +10%"
    failures, _ = check_regressions(_report([_case()]), _report([_case(peak_rss_kb=1101)]))
    assert "+10.1%" in failures[0]
    assert "limit +10.0%" in failures[0]
    failures, _ = check_regressions(
        _report([_case(amg_iterations=25)]), _report([_case(amg_iterations=28)])
    )
    assert "limit 27.5" in failures[0]
