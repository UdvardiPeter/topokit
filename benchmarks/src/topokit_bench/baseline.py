# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Read and validate the committed perf baseline.

Lives in the package, not in ``scripts/bench.py``, so it is importable and unit
tested under the same regime as the comparator: a bad baseline this loader lets
through is a gate that passes on numbers nobody generated, and catching that
should not require running a benchmark.

Two classes of file are rejected. First, shapes the comparator cannot read at
all: the baseline is committed, so it can be hand-edited or truncated in a merge,
and without these checks ``check_regressions`` raises a bare ``TypeError`` or
``AttributeError`` on a ``null`` case, a scalar ``meta`` or a string metric,
which reads as a harness bug rather than a bad file. Second, shapes the
comparator reads as *fewer gates* rather than as an error: it skips a case
carrying an ``error`` field with a note, and drops an unlabelled one from its
index, so a degraded baseline installed by following the regeneration procedure
would quietly gate less than it looks like it does.

Rejections raise ``SystemExit`` carrying the reason and a regeneration pointer;
the only caller is a command-line entry point that should print that and stop.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

METRICS = ("wall_per_iter_s", "peak_rss_kb", "amg_iterations")


def _bad_baseline(path: Path, reason: str) -> SystemExit:
    return SystemExit(f"{path}: {reason}; regenerate it (see benchmarks/README.md)")


def _check_case(path: Path, position: int, case: dict[str, Any]) -> None:
    """Reject one baseline case the comparator would skip over or trip on."""
    if "error" in case:
        # check_regressions treats an errored baseline entry as "no baseline for
        # this case" and only notes it, so this gates nothing while looking full.
        raise _bad_baseline(
            path,
            f"case {case.get('label')!r} carries an error field: a baseline must not carry "
            "a failed or unlabelled case",
        )
    if "label" not in case:
        # The comparator's index is keyed by label, so an unlabelled case is
        # simply absent from the gate; only the run side is checked for this.
        raise _bad_baseline(
            path,
            f"case #{position} has no label: a baseline must not carry a failed or unlabelled case",
        )
    label = case["label"]
    if not isinstance(label, str):
        # Labels key the comparator's index, so a list or dict is unhashable there.
        raise _bad_baseline(path, f"case label {label!r} is not a string")
    for field in METRICS:
        value = case.get(field)
        if value is not None and not isinstance(value, int | float):
            raise _bad_baseline(path, f"case {label!r} has non-numeric {field} {value!r}")


def load_baseline(path: Path) -> dict[str, Any]:
    """Return the baseline at ``path``, or exit nonzero saying why it is unusable."""
    try:
        data: Any = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise _bad_baseline(path, f"not valid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise _bad_baseline(path, f"top level is {type(data).__name__}, expected an object")
    meta = data.get("meta")
    if meta is not None and not isinstance(meta, dict):
        # The comparator reads meta first, so a scalar here fails before any case.
        raise _bad_baseline(path, f"'meta' is {type(meta).__name__}, expected an object")
    cases = data.get("cases", [])
    if not isinstance(cases, list) or any(not isinstance(case, dict) for case in cases):
        raise _bad_baseline(path, "'cases' is not a list of objects")
    for position, case in enumerate(cases):
        _check_case(path, position, case)
    return data
