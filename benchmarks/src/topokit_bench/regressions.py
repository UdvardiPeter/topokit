# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Compare a fresh benchmark run against the committed baseline.

Pure comparison over the report dicts ``scripts/bench.py`` writes, so the gate
logic is testable without running a benchmark. Tolerances are tiered by how
reproducible each metric is on the shared CI runner: peak RSS and AMG iteration
counts are near-deterministic and carry tight gates, wall time varies with CPU
contention and catches only gross regressions.
"""

from __future__ import annotations

from typing import Any

WALL_TOLERANCE = 0.30
RSS_TOLERANCE = 0.10
AMG_TOLERANCE = 0.10
AMG_ABS_SLACK = 2


def _index(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["label"]: c for c in report.get("cases", []) if "label" in c}


def check_regressions(
    baseline: dict[str, Any], latest: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Return ``(failures, notes)`` comparing ``latest`` against ``baseline``.

    Failures gate the nightly. Notes are informational: a disabled gate, a case
    present on only one side, or a metric neither report carries. A platform
    mismatch fails immediately without comparing cases, because numbers from
    different hardware are not comparable and a silently skipped gate is worse
    than a loud one.
    """
    failures: list[str] = []
    notes: list[str] = []
    base_meta = baseline.get("meta", {})
    new_meta = latest.get("meta", {})
    for key in ("system", "machine"):
        if base_meta.get(key) != new_meta.get(key):
            failures.append(
                f"baseline {key} {base_meta.get(key)!r} != run {key} {new_meta.get(key)!r}; "
                "regenerate the baseline on this platform"
            )
    if failures:
        return failures, notes

    amg_comparable = base_meta.get("pyamg") == new_meta.get("pyamg")
    if not amg_comparable:
        notes.append(
            f"pyamg {base_meta.get('pyamg')} to {new_meta.get('pyamg')}: AMG iteration "
            "gate skipped, counts are version sensitive"
        )
    base_cases = _index(baseline)
    new_cases = _index(latest)
    for label, new in sorted(new_cases.items()):
        if "error" in new:
            failures.append(f"{label}: case failed to run: {str(new['error'])[:200]}")
            continue
        base = base_cases.get(label)
        if base is None or "error" in base:
            notes.append(f"{label}: no baseline entry, not gated")
            continue
        for field, tol in (
            ("wall_per_iter_s", WALL_TOLERANCE),
            ("peak_rss_kb", RSS_TOLERANCE),
        ):
            old, now = base.get(field), new.get(field)
            if not old or now is None:
                notes.append(f"{label}: {field} missing on one side, not gated")
                continue
            if now > old * (1.0 + tol):
                failures.append(
                    f"{label}: {field} {now:.4g} vs baseline {old:.4g} "
                    f"({now / old - 1.0:+.0%}, limit {tol:+.0%})"
                )
        old_amg, new_amg = base.get("amg_iterations"), new.get("amg_iterations")
        if amg_comparable and old_amg and new_amg:
            limit = max(old_amg * (1.0 + AMG_TOLERANCE), old_amg + AMG_ABS_SLACK)
            if new_amg > limit:
                failures.append(
                    f"{label}: amg_iterations {new_amg} vs baseline {old_amg} (limit {limit:.0f})"
                )
    for label in sorted(set(base_cases) - set(new_cases)):
        notes.append(f"{label}: in the baseline but not this run")
    return failures, notes
