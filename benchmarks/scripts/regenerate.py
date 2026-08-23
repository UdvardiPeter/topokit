# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Regenerate the frozen reference fields for the benchmark suites.

Covers the per-PR 2D references and the nightly 3D + Michell references; ``--only
{2d,full,all}`` limits which set is rewritten, and ``--case SUBSTRING`` further
limits it to reference files whose name contains the substring (e.g. ``--only
full --case cantilever_tet`` rewrites only the tet4/tet10 nightly references).

MAINTAINER ONLY. Never run in CI. Run deliberately and commit the resulting
``tests/data/*.npz`` with a changelog note explaining the change. The suite
asserts against these frozen references; regenerating them resets the baseline.

Prefer regenerating on the CI platform (Linux): the suite tolerances absorb
cross-platform float drift, but a reference generated where CI runs leaves the
full tolerance as margin for genuine numerical changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from topokit.problem import Result, Schedule, Study

from topokit_bench.problems import BUILDERS, CASES, FULL_CASES, make_optimizer

DATA = Path(__file__).resolve().parent.parent / "tests" / "data"


def _freeze(path: Path, result: Result) -> None:
    np.savez(
        path,
        density=result.design.values,
        compliance=np.float64(result.objective),
        volume=np.float64(result.history["volume"][-1]),
        iterations=np.int64(result.iterations),
    )
    print(f"{path.name}: c={result.objective:.4f} it={result.iterations}")


def regenerate_2d(case_filter: str | None = None) -> None:
    """The per-PR Tier-3 2D references; ``case_filter`` limits by filename substring."""
    for name, nelx, nely, opt_name in CASES:
        filename = f"{name}_{nelx}x{nely}_{opt_name}.npz"
        if case_filter and case_filter not in filename:
            continue
        problem = BUILDERS[name](nelx, nely, optimizer=make_optimizer(opt_name))
        result = Study(problem, schedule=Schedule.single(p=3.0, max_iter=100, tol=1e-3)).run()
        _freeze(DATA / filename, result)


def regenerate_full(case_filter: str | None = None) -> None:
    """The nightly 3D + Michell references; ``case_filter`` limits by filename substring."""
    for case in FULL_CASES:
        for opt in case.optimizers:
            filename = f"{case.key}_{opt}.npz"
            if case_filter and case_filter not in filename:
                continue
            problem = case.build(**case.kwargs, optimizer=make_optimizer(opt))
            result = Study(problem, schedule=Schedule.single(p=3.0, max_iter=100, tol=1e-3)).run()
            _freeze(DATA / filename, result)


def main() -> None:
    """Regenerate frozen references; ``--only`` and ``--case`` limit which files are written."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("2d", "full", "all"), default="all")
    parser.add_argument(
        "--case",
        default=None,
        help="only rewrite reference files whose name contains this substring",
    )
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if args.only in ("2d", "all"):
        regenerate_2d(args.case)
    if args.only in ("full", "all"):
        regenerate_full(args.case)


if __name__ == "__main__":
    main()
