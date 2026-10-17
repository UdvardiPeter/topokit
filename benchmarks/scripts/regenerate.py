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
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
from topokit.problem import Problem, Result, Schedule, Study

from topokit_bench.problems import BUILDERS, CASES, FULL_CASES, make_optimizer

DATA = Path(__file__).resolve().parent.parent / "tests" / "data"
# The schedule every reference is frozen under, and the one both regression
# suites re-run; recorded into each file so a schedule change cannot quietly
# leave a stale reference passing on a coarse tolerance.
SCHEDULE = Schedule.single(p=3.0, max_iter=100, tol=1e-3)


def _freeze(path: Path, result: Result, *, problem: Problem, optimizer: str) -> None:
    mesh = problem.model.mesh
    (stage,) = SCHEDULE.stages
    np.savez(
        path,
        density=result.design.values,
        compliance=np.float64(result.objective),
        volume=np.float64(result.history["volume"][-1]),
        iterations=np.int64(result.iterations),
        # Provenance. A frozen number says nothing on its own: a failure has to
        # be readable as "the code changed" rather than "the case did", and the
        # platform a reference was frozen on decides how much cross-platform
        # margin the tolerances still hold. References frozen before this was
        # added carry none of it (see README).
        element_kind=mesh.element_kind,
        n_elements=np.int64(mesh.n_elements),
        optimizer=optimizer,
        penal=np.float64(stage.p),
        max_iter=np.int64(stage.max_iter),
        tol=np.float64(stage.tol),
        platform=f"{platform.system()}-{platform.machine()}",
        topokit=version("topokit"),
    )
    print(f"{path.name}: c={result.objective:.4f} it={result.iterations}")


def regenerate_2d(case_filter: str | None = None) -> int:
    """Rewrite the per-PR Tier-3 2D references; return how many files were written."""
    written = 0
    for name, nelx, nely, opt_name in CASES:
        filename = f"{name}_{nelx}x{nely}_{opt_name}.npz"
        if case_filter and case_filter not in filename:
            continue
        problem = BUILDERS[name](nelx, nely, optimizer=make_optimizer(opt_name))
        result = Study(problem, schedule=SCHEDULE).run()
        _freeze(DATA / filename, result, problem=problem, optimizer=opt_name)
        written += 1
    return written


def regenerate_full(case_filter: str | None = None) -> int:
    """Rewrite the nightly 3D + Michell references; return how many files were written."""
    written = 0
    for case in FULL_CASES:
        for opt in case.optimizers:
            filename = f"{case.key}_{opt}.npz"
            if case_filter and case_filter not in filename:
                continue
            problem = case.build(**case.kwargs, optimizer=make_optimizer(opt))
            result = Study(problem, schedule=SCHEDULE).run()
            _freeze(DATA / filename, result, problem=problem, optimizer=opt)
            written += 1
    return written


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
    written = 0
    if args.only in ("2d", "all"):
        written += regenerate_2d(args.case)
    if args.only in ("full", "all"):
        written += regenerate_full(args.case)
    if written == 0:
        # A mistyped or wrong-set --case writes nothing. Exiting 0 would read
        # as a successful regeneration over an unchanged tree, so fail loud.
        raise SystemExit(
            f"nothing regenerated: --only {args.only} with --case {args.case!r} "
            "matched no reference file"
        )


if __name__ == "__main__":
    main()
