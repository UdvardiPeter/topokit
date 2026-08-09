# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Compare a fresh benchmark run against the committed baseline.

Pure comparison over the report dicts ``scripts/bench.py`` writes, so the gate
logic is testable without running a benchmark. Tolerances are tiered by how
reproducible each metric is on the shared CI runner: peak RSS and AMG iteration
counts are near-deterministic and carry tight gates, wall time varies with CPU
contention and catches only gross regressions.

The gate is built so that it cannot pass by comparing nothing, nor by comparing
the wrong thing: an unusable baseline, an unlabelled or duplicated case, a case
whose size or solver no longer matches the baseline's, a metric the run dropped,
and a run that gated no metric at all are all failures rather than silence.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal

WALL_TOLERANCE = 0.30
RSS_TOLERANCE = 0.10
AMG_TOLERANCE = 0.10
AMG_ABS_SLACK = 2


def _index(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["label"]: c for c in report.get("cases", []) if "label" in c}


def _structural_failures(baseline: dict[str, Any], latest: dict[str, Any]) -> list[str]:
    """Return failures for reports the per-case loop would silently skip over.

    ``_index`` drops an unlabelled case and keeps only the last of a duplicated
    label, so a worker that died before labelling its record, or a report built
    by appending the same case twice, would shrink the gate without saying so.
    """
    failures = []
    for position, case in enumerate(latest.get("cases", [])):
        if "label" not in case:
            failures.append(
                f"run case #{position} has no label and cannot be gated; "
                "regenerate the run or fix the harness"
            )
    for side, report in (("baseline", baseline), ("run", latest)):
        counts = Counter(c["label"] for c in report.get("cases", []) if "label" in c)
        failures.extend(
            f"{label}: appears {n} times in the {side}; labels must be unique"
            for label, n in sorted(counts.items())
            if n > 1
        )
    return failures


def _identity_failures(label: str, base: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Return failures for a case whose definition moved out from under the baseline.

    The metrics only mean something if both sides measured the same problem with
    the same solver. Shrink a case in ``bench.py``'s ``CASES`` and every metric
    "improves" against numbers for a bigger problem, with the nightly green; a
    solver silently falling back to Direct (pyamg missing) reads the same way.
    A field absent from both sides is not a mismatch; absent from one side is,
    since then one report describes work the other cannot vouch for.
    """
    failures = []
    for field in ("elements", "dof", "solver"):
        old, now = base.get(field), new.get(field)
        if old == now:
            continue
        failures.append(
            f"{label}: {field} {old!r} in the baseline but {now!r} in this run; "
            "the two reports describe different work, so the baseline must be "
            "regenerated"
        )
    return failures


def _uncomparable(
    label: str, field: str, old: Any, now: Any
) -> tuple[Literal["failure", "note"], str] | None:
    """Return ``("failure" | "note", message)`` when ``field`` cannot be compared.

    ``None`` means both sides carry a usable positive number, so the caller may
    compare them and divide by ``old``. Every other combination is reported: a
    metric only the baseline has is a regression in coverage (failure), while a
    metric only the run has, or one neither side has, is a stale baseline (note).
    """
    if old is None:
        if now is None:
            return "note", f"{label}: {field} absent from both, not gated"
        return "note", f"{label}: {field} not in the baseline, not gated"
    if not (old > 0):
        # Zero, negative or NaN: useless as a denominator and meaningless as a bound.
        # For all three metrics (wall time, RSS, AMG iterations), zero in a fresh run is
        # a deliberate hard failure: it signals a broken counter, not a valid measurement.
        # This is the old ``not old`` guard, kept so the comparison below can never raise
        # ZeroDivisionError while formatting its percentage.
        return "note", f"{label}: {field} baseline value {old!r} is not usable, not gated"
    if now is None:
        return "failure", (
            f"{label}: {field} present in the baseline ({old:.4g}) but missing from the run"
        )
    if not (now > 0):
        return "failure", f"{label}: {field} implausible value {now!r} (baseline {old:.4g})"
    return None


def check_regressions(
    baseline: dict[str, Any], latest: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Return ``(failures, notes)`` comparing ``latest`` against ``baseline``.

    Failures gate the nightly. Notes are informational: a disabled gate, a case
    present on only one side, or a metric neither report carries. A platform
    mismatch, or a platform either report failed to record, fails immediately
    without comparing cases, because numbers from different hardware are not
    comparable and a silently skipped gate is worse than a loud one. For the
    same reason a run that ends up comparing no metric at all is a failure, not
    a pass.
    """
    failures: list[str] = []
    notes: list[str] = []
    base_meta = baseline.get("meta") or {}
    new_meta = latest.get("meta") or {}
    for key in ("system", "machine"):
        base_value, new_value = base_meta.get(key), new_meta.get(key)
        if base_value is None or new_value is None:
            # Without this, two reports carrying no meta at all compared
            # None == None and gated every case on unknown hardware.
            failures.append(
                f"{key} missing: baseline {base_value!r}, run {new_value!r}; "
                "regenerate the baseline on this platform"
            )
        elif base_value != new_value:
            failures.append(
                f"baseline {key} {base_value!r} != run {key} {new_value!r}; "
                "regenerate the baseline on this platform"
            )
    if failures:
        return failures, notes

    failures.extend(_structural_failures(baseline, latest))
    amg_comparable = base_meta.get("pyamg") == new_meta.get("pyamg")
    if not amg_comparable:
        notes.append(
            f"pyamg {base_meta.get('pyamg')} to {new_meta.get('pyamg')}: AMG iteration "
            "gate skipped, counts are version sensitive"
        )
    base_cases = _index(baseline)
    new_cases = _index(latest)
    gated = 0
    for label, new in sorted(new_cases.items()):
        if "error" in new:
            failures.append(f"{label}: case failed to run: {str(new['error'])[:200]}")
            continue
        base = base_cases.get(label)
        if base is None or "error" in base:
            notes.append(f"{label}: no baseline entry, not gated")
            continue
        identity = _identity_failures(label, base, new)
        if identity:
            # Comparing metrics across two different problems is meaningless, so
            # skip them; not incrementing ``gated`` keeps this off the coverage
            # count, which is what makes a wholly redefined run fail twice over.
            failures.extend(identity)
            continue
        for field, tol in (
            ("wall_per_iter_s", WALL_TOLERANCE),
            ("peak_rss_kb", RSS_TOLERANCE),
        ):
            # annotated Any, not the inferred ``Any | None``: _uncomparable has
            # already rejected None by the time either is used as a number
            old: Any = base.get(field)
            now: Any = new.get(field)
            verdict = _uncomparable(label, field, old, now)
            if verdict is not None:
                (failures if verdict[0] == "failure" else notes).append(verdict[1])
                continue
            gated += 1
            if now > old * (1.0 + tol):
                failures.append(
                    f"{label}: {field} {now:.4g} vs baseline {old:.4g} "
                    f"({now / old - 1.0:+.1%}, limit {tol:+.1%})"
                )
        if not amg_comparable:
            continue  # the pyamg note above already accounts for the skip
        old_amg: Any = base.get("amg_iterations")
        new_amg: Any = new.get("amg_iterations")
        if old_amg is None and new_amg is None:
            # A 2D direct-solver case legitimately has no AMG count on either
            # side, so unlike wall time and RSS that is not worth a note.
            continue
        verdict = _uncomparable(label, "amg_iterations", old_amg, new_amg)
        if verdict is not None:
            (failures if verdict[0] == "failure" else notes).append(verdict[1])
            continue
        gated += 1
        limit = max(old_amg * (1.0 + AMG_TOLERANCE), old_amg + AMG_ABS_SLACK)
        if new_amg > limit:
            failures.append(
                f"{label}: amg_iterations {new_amg} vs baseline {old_amg} (limit {limit:.1f})"
            )
    for label in sorted(set(base_cases) - set(new_cases)):
        notes.append(f"{label}: in the baseline but not this run")
    if gated == 0:
        failures.append("no metric was compared against the baseline; the run is not gated")
    return failures, notes
