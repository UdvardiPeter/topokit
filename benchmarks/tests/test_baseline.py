# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Unit tests for the committed-baseline loader (no benchmark run needed)."""

import json
from pathlib import Path
from typing import Any

import pytest

from topokit_bench.baseline import load_baseline

COMMITTED = Path(__file__).resolve().parents[1] / "bench" / "baseline.json"


def _case(**over: Any) -> dict[str, Any]:
    return {
        "label": "cantilever_3d_20",
        "wall_per_iter_s": 1.0,
        "peak_rss_kb": 1000,
        "amg_iterations": 20,
    } | over


def _report(cases: list[Any]) -> dict[str, Any]:
    # list[Any]: several fixtures below are deliberately malformed
    return {"meta": {"system": "Linux", "machine": "x86_64"}, "cases": cases}


def _write(tmp_path: Path, data: Any) -> Path:
    path = tmp_path / "baseline.json"
    path.write_text(data if isinstance(data, str) else json.dumps(data))
    return path


def test_the_committed_baseline_loads() -> None:
    # the real file the nightly gates against, not a fixture
    data = load_baseline(COMMITTED)
    assert data["meta"]["system"]
    assert data["meta"]["machine"]
    assert [case["label"] for case in data["cases"]]


def test_a_hand_written_baseline_loads(tmp_path: Path) -> None:
    # missing meta, empty cases and a null AMG count are all the comparator's
    # problem to fail on, not the loader's to reject
    assert load_baseline(_write(tmp_path, {"cases": []})) == {"cases": []}
    data = load_baseline(_write(tmp_path, _report([_case(amg_iterations=None)])))
    assert data["cases"][0]["amg_iterations"] is None


@pytest.mark.parametrize(
    "data,expected",
    [
        ("{ truncated in a merge", "not valid JSON"),
        ([_case()], "top level is list"),
        ({"meta": "Linux", "cases": []}, "'meta' is str"),
        ({"cases": {"a": 1}}, "'cases' is not a list of objects"),
        (_report([_case(), None]), "'cases' is not a list of objects"),
        # the comparator only skips an errored or unlabelled baseline case, so
        # one would gate nothing while the file looks full
        (_report([{"label": "cantilever_3d_20", "error": "boom"}]), "case #0"),
        (_report([_case(), {"wall_per_iter_s": 1.0}]), "case #1"),
        (_report([_case(label=["a", "b"])]), "case #0"),
        (_report([_case(peak_rss_kb="215264")]), "non-numeric peak_rss_kb"),
    ],
)
def test_an_unusable_baseline_is_rejected(tmp_path: Path, data: Any, expected: str) -> None:
    with pytest.raises(SystemExit) as exc:
        load_baseline(_write(tmp_path, data))
    message = str(exc.value)
    assert expected in message
    assert "regenerate it (see benchmarks/README.md)" in message
