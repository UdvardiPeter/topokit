# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Compare a fresh benchmark run against the committed baseline.

Pure comparison over the report dicts ``scripts/bench.py`` writes, so the gate
logic is testable without running a benchmark. Tolerances are tiered by how
reproducible each metric is on the shared CI runner: peak RSS and AMG iteration
counts are near-deterministic and carry tight gates, wall time varies with CPU
contention and catches only gross regressions.

The gate cannot pass by comparing nothing, nor by comparing the wrong thing: a
platform mismatch, a case whose size or solver no longer matches the baseline's,
a metric the run dropped, and a run that gated no metric at all are failures.
Everything else is silence, so every line this returns is a number that moved.
"""

from __future__ import annotations

from typing import Any

WALL_TOLERANCE = 0.30
RSS_TOLERANCE = 0.10
AMG_TOLERANCE = 0.10
AMG_ABS_SLACK = 2


def _index(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index a report's cases by label; an unlabelled case cannot be gated."""
    return {case["label"]: case for case in report.get("cases", []) if "label" in case}


def check_regressions(baseline: dict[str, Any], latest: dict[str, Any]) -> list[str]:
    """Return the regressions of ``latest`` against ``baseline``, empty if clean."""
    base_meta: dict[str, Any] = baseline.get("meta") or {}
    new_meta: dict[str, Any] = latest.get("meta") or {}
    # Numbers from different hardware are not comparable, so a mismatch stops
    # before any case. A key missing on either side counts as a mismatch, or two
    # reports carrying no meta at all would pass on None == None.
    platform = [
        f"{key}: baseline {base_meta.get(key)!r} != run {new_meta.get(key)!r}; "
        "regenerate the baseline on this platform"
        for key in ("system", "machine")
        if base_meta.get(key) is None or base_meta.get(key) != new_meta.get(key)
    ]
    if platform:
        return platform

    # AMG iteration counts are pyamg-version sensitive, so a version change
    # disables that gate rather than reporting a regression nobody caused.
    amg_comparable = base_meta.get("pyamg") == new_meta.get("pyamg")
    base_cases = _index(baseline)
    failures: list[str] = []
    gated = 0
    for label, new in sorted(_index(latest).items()):
        if "error" in new:
            failures.append(f"{label}: case failed to run: {str(new['error'])[:200]}")
            continue
        base = base_cases.get(label)
        if base is None:
            continue  # a case the baseline predates; nothing to compare it against
        identity = [
            f"{label}: {field} {base.get(field)!r} in the baseline but {new.get(field)!r} "
            "in this run; the two reports describe different work, so the baseline "
            "must be regenerated"
            for field in ("elements", "dof", "solver")
            if base.get(field) != new.get(field)
        ]
        if identity:
            # Comparing metrics across two different problems is meaningless, so
            # skip them; leaving them out of ``gated`` is what makes a wholly
            # redefined run fail twice over.
            failures.extend(identity)
            continue
        gates: list[tuple[str, float, int]] = [
            ("wall_per_iter_s", WALL_TOLERANCE, 0),
            ("peak_rss_kb", RSS_TOLERANCE, 0),
        ]
        if amg_comparable:
            gates.append(("amg_iterations", AMG_TOLERANCE, AMG_ABS_SLACK))
        for field, tol, slack in gates:
            # Any, not the inferred ``Any | None``: both are checked for None
            # before either is used as a number.
            old: Any = base.get(field)
            now: Any = new.get(field)
            if old is None or not (old > 0):
                continue  # no usable bound: an absent metric, or a broken counter
            if now is None or not (now > 0):
                # A metric the baseline gates and the run dropped is a loss of
                # coverage, not a stale baseline: fail rather than note it.
                failures.append(
                    f"{label}: {field} present in the baseline ({old:.4g}) but {now!r} in the run"
                )
                continue
            gated += 1
            limit = max(old * (1.0 + tol), old + slack)
            if now > limit:
                failures.append(
                    f"{label}: {field} {now:.4g} vs baseline {old:.4g} "
                    f"({now / old - 1.0:+.1%}, limit {limit:.4g})"
                )
    if gated == 0:
        failures.append("no metric was compared against the baseline; the run is not gated")
    return failures
