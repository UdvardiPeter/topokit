"""Tier-3 full 3D + Michell reference regression (pytest -m regression_full)."""

from pathlib import Path

import numpy as np
import pytest
from topokit.problem import Result, Schedule, Study

from topokit_bench.problems import FULL_CASES, FullCase, make_optimizer

DATA = Path(__file__).resolve().parent / "data"

_RUNS = [(case, opt) for case in FULL_CASES for opt in case.optimizers]
_IDS = [f"{case.key}_{opt}" for case, opt in _RUNS]


def _run(case: FullCase, opt: str) -> Result:
    problem = case.build(**case.kwargs, optimizer=make_optimizer(opt))
    return Study(problem, schedule=Schedule.single(p=3.0, max_iter=100, tol=1e-3)).run()


@pytest.mark.regression_full
@pytest.mark.parametrize("case,opt", _RUNS, ids=_IDS)
def test_matches_reference(case: FullCase, opt: str) -> None:
    ref = np.load(DATA / f"{case.key}_{opt}.npz")
    result = _run(case, opt)
    assert result.history["volume"][-1] == pytest.approx(float(ref["volume"]), abs=1e-3)
    assert result.objective == pytest.approx(float(ref["compliance"]), rel=0.01)
    assert float(np.mean(np.abs(result.design.values - ref["density"]))) < 1e-2
    assert abs(result.iterations - int(ref["iterations"])) <= 0.25 * int(ref["iterations"]) + 1


@pytest.mark.regression_full
def test_michell_oc_and_mma_agree() -> None:
    # the two optimizers reach the same Michell optimum (checked on the frozen
    # refs, mirroring the 2D suite's agreement gate).
    c_oc = float(np.load(DATA / "michell_90x30_oc.npz")["compliance"])
    c_mma = float(np.load(DATA / "michell_90x30_mma.npz")["compliance"])
    assert abs(c_mma - c_oc) / c_oc < 0.05


@pytest.mark.regression_full
def test_michell_reference_is_vertically_symmetric() -> None:
    # topology-sanity: the committed Michell reference is mirror-symmetric about
    # the horizontal centreline (symmetric BCs + load).
    ref = np.load(DATA / "michell_90x30_oc.npz")
    field = ref["density"].reshape(30, 90)  # (nely, nelx), x-fastest
    assert float(np.mean(np.abs(field - field[::-1, :]))) < 1e-2
