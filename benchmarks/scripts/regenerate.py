# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Regenerate the frozen reference fields for the 2D benchmarks.

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


def regenerate_2d() -> None:
    """The per-PR Tier-3 2D references."""
    for name, nelx, nely, opt_name in CASES:
        problem = BUILDERS[name](nelx, nely, optimizer=make_optimizer(opt_name))
        result = Study(problem, schedule=Schedule.single(p=3.0, max_iter=100, tol=1e-3)).run()
        _freeze(DATA / f"{name}_{nelx}x{nely}_{opt_name}.npz", result)


def regenerate_full() -> None:
    """The nightly 3D + Michell references."""
    for case in FULL_CASES:
        for opt in case.optimizers:
            problem = case.build(**case.kwargs, optimizer=make_optimizer(opt))
            result = Study(problem, schedule=Schedule.single(p=3.0, max_iter=100, tol=1e-3)).run()
            _freeze(DATA / f"{case.key}_{opt}.npz", result)


def main() -> None:
    """Regenerate frozen references; ``--only`` limits which set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("2d", "full", "all"), default="all")
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if args.only in ("2d", "all"):
        regenerate_2d()
    if args.only in ("full", "all"):
        regenerate_full()


if __name__ == "__main__":
    main()
