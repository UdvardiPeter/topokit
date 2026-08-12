# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Unit tests for the committed-baseline loader (no benchmark run needed)."""

import json
from pathlib import Path
from typing import Any

import pytest

from topokit_bench.baseline import load_baseline

COMMITTED = Path(__file__).resolve().parents[1] / "bench" / "baseline.json"


def _write(tmp_path: Path, data: Any) -> Path:
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(data))
    return path


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


def _report(cases: list[Any]) -> dict[str, Any]:
    # list[Any]: several fixtures below are deliberately malformed
    return {"meta": {"system": "Linux", "machine": "x86_64", "pyamg": "5.3.0"}, "cases": cases}


def _rejects(tmp_path: Path, data: Any) -> str:
    with pytest.raises(SystemExit) as exc:
        load_baseline(_write(tmp_path, data))
    message = str(exc.value)
    assert "regenerate it (see benchmarks/README.md)" in message
    return message


def test_the_committed_baseline_loads() -> None:
    # the real file the nightly gates against, not a fixture
    data = load_baseline(COMMITTED)
    assert data["meta"]["system"]
    assert data["meta"]["machine"]
    assert [case["label"] for case in data["cases"]]


def test_a_hand_written_baseline_loads(tmp_path: Path) -> None:
    data = load_baseline(_write(tmp_path, _report([_case()])))
    assert data["cases"][0]["label"] == "cantilever_3d_20"


def test_missing_meta_and_empty_cases_load(tmp_path: Path) -> None:
    # both are the comparator's problem to fail on, not the loader's to reject
    assert load_baseline(_write(tmp_path, {"cases": []})) == {"cases": []}


def test_invalid_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    path.write_text("{ truncated in a merge")
    with pytest.raises(SystemExit) as exc:
        load_baseline(path)
    assert "not valid JSON" in str(exc.value)


def test_non_object_top_level_is_rejected(tmp_path: Path) -> None:
    assert "top level is list" in _rejects(tmp_path, [_case()])


def test_scalar_meta_is_rejected(tmp_path: Path) -> None:
    assert "'meta' is str" in _rejects(tmp_path, {"meta": "Linux", "cases": []})


def test_non_list_cases_is_rejected(tmp_path: Path) -> None:
    assert "'cases' is not a list of objects" in _rejects(tmp_path, {"cases": {"a": 1}})


def test_null_case_is_rejected(tmp_path: Path) -> None:
    assert "'cases' is not a list of objects" in _rejects(tmp_path, _report([_case(), None]))


def test_errored_case_is_rejected(tmp_path: Path) -> None:
    # check_regressions only notes an errored baseline entry, so it would gate nothing
    message = _rejects(tmp_path, _report([{"label": "cantilever_3d_20", "error": "boom"}]))
    assert "carries an error field" in message
    assert "must not carry a failed or unlabelled case" in message


def test_unlabelled_case_is_rejected(tmp_path: Path) -> None:
    # the comparator checks the run side for this; the baseline side is ours
    message = _rejects(tmp_path, _report([_case(), {"wall_per_iter_s": 1.0}]))
    assert "case #1 has no label" in message
    assert "must not carry a failed or unlabelled case" in message


def test_non_string_label_is_rejected(tmp_path: Path) -> None:
    assert "is not a string" in _rejects(tmp_path, _report([_case(label=["a", "b"])]))


def test_non_numeric_metric_is_rejected(tmp_path: Path) -> None:
    message = _rejects(tmp_path, _report([_case(peak_rss_kb="215264")]))
    assert "non-numeric peak_rss_kb" in message


def test_null_metrics_are_allowed(tmp_path: Path) -> None:
    # a 2D direct-solver case legitimately carries no AMG count
    data = load_baseline(_write(tmp_path, _report([_case(amg_iterations=None)])))
    assert data["cases"][0]["amg_iterations"] is None
