# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Read and validate the committed perf baseline.

Lives in the package rather than in ``scripts/bench.py`` so a bad baseline is
caught by a unit test instead of by a nightly that passes on numbers nobody
generated. The file is committed, so it can be hand-edited or truncated in a
merge: every shape ``check_regressions`` would trip on, or would read as *fewer
gates* rather than as an error, is rejected here.

Rejections raise ``SystemExit`` carrying the reason and a regeneration pointer;
the only caller is a command-line entry point that should print that and stop.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

METRICS = ("wall_per_iter_s", "peak_rss_kb", "amg_iterations")


def _reject(path: Path, reason: str) -> SystemExit:
    return SystemExit(f"{path}: {reason}; regenerate it (see benchmarks/README.md)")


def load_baseline(path: Path) -> dict[str, Any]:
    """Return the baseline at ``path``, or exit nonzero saying why it is unusable."""
    try:
        data: Any = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise _reject(path, f"not valid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise _reject(path, f"top level is {type(data).__name__}, expected an object")
    meta = data.get("meta")
    if meta is not None and not isinstance(meta, dict):
        # The comparator reads meta first, so a scalar here fails before any case.
        raise _reject(path, f"'meta' is {type(meta).__name__}, expected an object")
    cases = data.get("cases", [])
    if not isinstance(cases, list) or any(not isinstance(case, dict) for case in cases):
        raise _reject(path, "'cases' is not a list of objects")
    for position, case in enumerate(cases):
        # The comparator skips an errored baseline entry and indexes by label, so
        # either shape gates nothing while looking full.
        if "error" in case or not isinstance(case.get("label"), str):
            raise _reject(
                path,
                f"case #{position} carries an error field or has no string label: "
                "a baseline must not carry a failed or unlabelled case",
            )
        for field in METRICS:
            value = case.get(field)
            if value is not None and not isinstance(value, int | float):
                raise _reject(path, f"case {case['label']!r} has non-numeric {field} {value!r}")
    return data
