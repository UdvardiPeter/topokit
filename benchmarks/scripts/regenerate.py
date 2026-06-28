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

from pathlib import Path

import numpy as np
from topokit.problem import Schedule, Study

from topokit_bench.problems import BUILDERS, CASES, make_optimizer

DATA = Path(__file__).resolve().parent.parent / "tests" / "data"


def main() -> None:
    """Run every case and write its reference .npz."""
    DATA.mkdir(parents=True, exist_ok=True)
    for name, nelx, nely, opt_name in CASES:
        problem = BUILDERS[name](nelx, nely, optimizer=make_optimizer(opt_name))
        # compliance is converged by ~iter 100 (within 0.1% of iter 300) while OC's
        # design-change criterion never trips, so cap there: literature-grade
        # compliance, ~3x faster, keeps the per-PR Tier-3 suite under budget.
        result = Study(problem, schedule=Schedule.single(p=3.0, max_iter=100, tol=1e-3)).run()
        path = DATA / f"{name}_{nelx}x{nely}_{opt_name}.npz"
        np.savez(
            path,
            density=result.design.values,
            compliance=np.float64(result.objective),
            volume=np.float64(result.history["volume"][-1]),
            iterations=np.int64(result.iterations),
        )
        print(f"{path.name}: c={result.objective:.4f} it={result.iterations}")


if __name__ == "__main__":
    main()
