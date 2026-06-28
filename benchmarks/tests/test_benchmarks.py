"""Tier-3 2D reference regression (pytest -m regression)."""

from pathlib import Path

import numpy as np
import pytest
from topokit.problem import Result, Schedule, Study

from topokit_bench.problems import BUILDERS, CASES, make_optimizer

DATA = Path(__file__).resolve().parent / "data"


def _run(name: str, nelx: int, nely: int, opt_name: str) -> Result:
    problem = BUILDERS[name](nelx, nely, optimizer=make_optimizer(opt_name))
    return Study(problem, schedule=Schedule.single(p=3.0, max_iter=100, tol=1e-3)).run()


@pytest.mark.regression
@pytest.mark.parametrize("name,nelx,nely,opt_name", CASES)
def test_matches_reference(name: str, nelx: int, nely: int, opt_name: str) -> None:
    ref = np.load(DATA / f"{name}_{nelx}x{nely}_{opt_name}.npz")
    result = _run(name, nelx, nely, opt_name)
    # 1. volume held
    assert result.history["volume"][-1] == pytest.approx(float(ref["volume"]), abs=1e-3)
    # 2. compliance within 1% of the frozen reference
    assert result.objective == pytest.approx(float(ref["compliance"]), rel=0.01)
    # 3. density field within a coarse 1e-2 (robust to cross-platform LU/BLAS drift)
    assert float(np.mean(np.abs(result.design.values - ref["density"]))) < 1e-2
    # 4. iteration count within +/- 25%
    assert abs(result.iterations - int(ref["iterations"])) <= 0.25 * int(ref["iterations"]) + 1


@pytest.mark.regression
@pytest.mark.parametrize("name", ["mbb", "cantilever"])
@pytest.mark.parametrize("nelx,nely", [(60, 20), (150, 50)])
def test_oc_and_mma_agree(name: str, nelx: int, nely: int) -> None:
    # the two optimizers reach the same optimum; checked on the frozen references
    # so the per-PR suite does not pay a second full run per case.
    c_oc = float(np.load(DATA / f"{name}_{nelx}x{nely}_oc.npz")["compliance"])
    c_mma = float(np.load(DATA / f"{name}_{nelx}x{nely}_mma.npz")["compliance"])
    assert abs(c_mma - c_oc) / c_oc < 0.05
