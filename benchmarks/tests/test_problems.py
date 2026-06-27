import numpy as np
import pytest

from topokit.optimizers import OC
from topokit.problem import Schedule, Study
from topokit_bench.problems import cantilever, mbb


@pytest.mark.parametrize("build", [mbb, cantilever])
def test_builder_makes_runnable_problem(build) -> None:
    p = build(20, 10, optimizer=OC(move=0.2))
    assert p.model.n_dof > 0
    assert p.chain.n_vars == 20 * 10  # all elements are design
    p.solver.prepare(p.model.assemble(p.chain.apply(p.chain.initial_design(0.5))))
    u = np.asarray(p.solver.solve(p.model.loads()))
    assert np.isfinite(u).all() and np.abs(u).max() > 0.0


def test_mbb_supports_are_nonempty() -> None:
    p = mbb(20, 10, optimizer=OC(move=0.2))
    assert p.model.n_dof < 2 * 21 * 11  # some DOFs fixed


@pytest.mark.parametrize("build,volfrac", [(mbb, 0.5), (cantilever, 0.4)])
def test_optimization_produces_sane_design(build, volfrac) -> None:
    p = build(60, 20, optimizer=OC(move=0.2))
    result = Study(p, schedule=Schedule.single(p=3.0, max_iter=120, tol=1e-3)).run()
    rho = result.design.values
    assert result.history["volume"][-1] == pytest.approx(volfrac, abs=1e-3)
    assert result.objective < 0.6 * result.history["objective"][0]
    # single-stage density filtering (no Heaviside) leaves grey boundaries by
    # design; the uniform initial is 100% grey, so this confirms real movement
    # toward 0/1 without demanding the crisp result projection would give.
    grey = float(np.mean((rho > 0.1) & (rho < 0.9)))
    assert grey < 0.6
