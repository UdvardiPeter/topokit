from collections.abc import Callable

import numpy as np
import pytest
from topokit.fem import LinearElasticity, Material, PointLoad
from topokit.mesh import StructuredGrid
from topokit.optimizers import OC
from topokit.problem import Problem, Schedule, Study
from topokit.selection import NearPoint, PlaneSlab

from topokit_bench.problems import cantilever, mbb

Builder = Callable[..., Problem]


def test_element_stiffness_matches_88line() -> None:
    # The lineage anchor's foundation: TopoKit's Q4 plane-stress element
    # reproduces the 88-line KE (Andreassen et al. 2011) exactly, so a faithful
    # single-stage run reproduces the 88-line method. Eigenvalues are
    # DOF-ordering-independent, so this is an exact, platform-independent check.
    g = StructuredGrid.box(size=(2.0, 2.0), shape=(2, 2))
    model = LinearElasticity(
        g,
        Material(E=1.0, nu=0.3, rho=1.0),
        supports=[(PlaneSlab((0.0, 0.0), (1.0, 0.0), tol=1e-9), "all")],
        loads=[PointLoad(NearPoint((2.0, 2.0)), (0.0, -1.0))],
    )
    nu = 0.3
    a11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]])
    a12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]])
    b11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]])
    b12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]])
    ke = (
        1.0
        / (1.0 - nu**2)
        / 24.0
        * (np.block([[a11, a12], [a12.T, a11]]) + nu * np.block([[b11, b12], [b12.T, b11]]))
    )
    np.testing.assert_allclose(
        np.linalg.eigvalsh(model.element_stiffness), np.linalg.eigvalsh(ke), atol=1e-12
    )


@pytest.mark.parametrize("build", [mbb, cantilever])
def test_builder_makes_runnable_problem(build: Builder) -> None:
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
def test_optimization_produces_sane_design(build: Builder, volfrac: float) -> None:
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


def test_cantilever_3d_builds_and_runs() -> None:
    from topokit.problem import Schedule, Study

    from topokit_bench.problems import cantilever_3d, make_optimizer

    problem = cantilever_3d(8, 4, 4, optimizer=make_optimizer("oc"))
    assert problem.model.mesh.dim == 3
    assert problem.model.mesh.element_kind == "hex8"
    result = Study(problem, schedule=Schedule.single(p=3.0, max_iter=5, tol=1e-3)).run()
    assert np.all(np.isfinite(result.design.values))
    assert result.history["volume"][-1] == pytest.approx(0.3, abs=1e-3)


def test_michell_builds_and_is_symmetric() -> None:
    from topokit.problem import Schedule, Study

    from topokit_bench.problems import make_optimizer, michell

    problem = michell(30, 12, optimizer=make_optimizer("oc"))
    assert problem.model.mesh.dim == 2
    result = Study(problem, schedule=Schedule.single(p=3.0, max_iter=20, tol=1e-3)).run()
    field = result.design.values.reshape(12, 30)  # (nely, nelx), x-fastest
    assert float(np.mean(np.abs(field - field[::-1, :]))) < 2e-2
