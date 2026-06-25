"""Tests for the Problem/Study orchestration layer."""

import numpy as np
import pytest

from topokit.events import FieldSnapshot, IterationFinished, StudyFinished, StudyStarted
from topokit.fem import LinearElasticity, Material, PointLoad
from topokit.fields import FieldSpec
from topokit.mesh import StructuredGrid
from topokit.optimizers import MMA, OC, Optimizer
from topokit.parametrization import SIMP, DensityFilter
from topokit.problem import Problem, ProblemError, Study
from topokit.responses import Compliance, Volume
from topokit.selection import Box, PlaneSlab
from topokit.solvers import AmgCG, Direct, LinearSolver


def _cantilever(shape: tuple[int, int] = (20, 10)) -> LinearElasticity:
    g = StructuredGrid.box(size=(float(shape[0]), float(shape[1])), shape=shape)
    left = PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0), tol=1e-9)
    tip = Box((float(shape[0]), 0.0), (float(shape[0]), 0.0), tol=0.6)
    return LinearElasticity(
        g,
        Material(E=1.0, nu=0.3, rho=1.0),
        supports=[(left, "all")],
        loads=[PointLoad(tip, (0.0, -1.0))],
    )


def _problem(optimizer: Optimizer | None = None, solver: LinearSolver | str = "auto") -> Problem:
    model = _cantilever()
    chain = DensityFilter(radius=1.5) | SIMP(p=3.0)
    return Problem(
        model,
        chain,
        objective=Compliance(),
        constraints=[Volume() <= 0.4],
        optimizer=optimizer or OC(move=0.2),
        solver=solver,
    )


def test_problem_validates_field_mismatch() -> None:
    model = _cantilever()

    class WrongTerminal(SIMP):
        pass

    # a chain whose terminal field does not match the physics expectation
    class FakePhysics:
        expected_field = FieldSpec("conductivity_scale")  # not stiffness_scale
        mesh = model.mesh

    with pytest.raises(ProblemError, match="field"):
        Problem(FakePhysics(), DensityFilter(radius=1.5) | SIMP(), objective=Compliance())  # type: ignore[arg-type]


def test_problem_binds_unbound_chain() -> None:
    p = _problem()
    assert p.chain.n_vars == p.model.mesh.design.sum()


def test_problem_solver_auto_resolves() -> None:
    small = _problem(solver="auto")
    assert isinstance(small.solver, Direct)  # cantilever is small / 2D
    explicit = _problem(solver=Direct())
    assert isinstance(explicit.solver, Direct)


def test_problem_unknown_solver_string_raises() -> None:
    with pytest.raises(ProblemError, match="solver"):
        _problem(solver="bogus")


def test_problem_accepts_prebound_chain() -> None:
    model = _cantilever()
    bound = (DensityFilter(radius=1.5) | SIMP(p=3.0)).bind(model.mesh)
    p = Problem(model, bound, objective=Compliance(), constraints=[Volume() <= 0.4])
    assert p.chain is bound  # used as-is, not re-bound


def test_duplicate_constraint_names_raise() -> None:
    # both Volume constraints report under "volume"; reporting would silently
    # drop one, so Problem rejects the collision and points at .labeled(...)
    with pytest.raises(ProblemError, match="label"):
        Problem(
            _cantilever(),
            DensityFilter(radius=1.5) | SIMP(p=3.0),
            objective=Compliance(),
            constraints=[Volume(region="design") <= 0.4, Volume(region="all") <= 0.6],
            optimizer=MMA(),
        )


def test_labeled_constraints_reported_separately() -> None:
    p = Problem(
        _cantilever(),
        DensityFilter(radius=1.5) | SIMP(p=3.0),
        objective=Compliance(),
        constraints=[
            (Volume(region="design") <= 0.4).labeled("vol_design"),
            (Volume(region="all") <= 0.6).labeled("vol_all"),
        ],
        optimizer=MMA(),
    )
    result = Study(p, max_iter=5, tol=0.0).run()
    assert len(result.history["vol_design"]) == 5
    assert len(result.history["vol_all"]) == 5


def test_study_reduces_compliance_with_oc() -> None:
    # OC plateaus on this grey (no-projection) cantilever rather than reaching a
    # tight change tol, so assert the optimization worked, not convergence.
    result = Study(_problem(OC(move=0.2)), max_iter=120, tol=1e-3).run()
    assert result.history["objective"][-1] < 0.5 * result.history["objective"][0]
    assert result.history["volume"][-1] == pytest.approx(0.4, abs=1e-3)


def test_study_with_mma_matches_oc() -> None:
    # MMA needs objective normalization (Study supplies it); without it MMA
    # would converge wrong. Assert MMA reaches the same compliance as OC.
    c_oc = Study(_problem(OC(move=0.2)), max_iter=120, tol=1e-3).run().objective
    c_mma = Study(_problem(MMA()), max_iter=120, tol=1e-3).run().objective
    assert abs(c_mma - c_oc) / c_oc < 0.05


def test_study_emits_events() -> None:
    study = Study(_problem(), max_iter=20, tol=0.0, snapshot_every=5)
    started: list[StudyStarted] = []
    iters: list[IterationFinished] = []
    snaps: list[FieldSnapshot] = []
    finished: list[StudyFinished] = []
    study.events.subscribe(StudyStarted, started.append)
    study.events.subscribe(IterationFinished, iters.append)
    study.events.subscribe(FieldSnapshot, snaps.append)
    study.events.subscribe(StudyFinished, finished.append)
    study.run()
    assert len(started) == 1
    assert len(finished) == 1
    assert len(iters) == 20
    assert len(snaps) == 4  # iters 5, 10, 15, 20
    assert "compliance" in iters[-1].responses
    assert "volume" in iters[-1].responses


def test_iterate_matches_run() -> None:
    states = list(Study(_problem(), max_iter=15, tol=0.0).iterate())
    assert len(states) == 15
    final_iter = Study(_problem(), max_iter=15, tol=0.0).run()
    np.testing.assert_allclose(states[-1].x, final_iter.x)


def test_convergence_stops_at_tol() -> None:
    # MMA reaches a tight change tol on the cantilever (~78 iters); OC plateaus
    result = Study(_problem(MMA()), max_iter=200, tol=1e-3).run()
    assert result.converged
    assert result.iterations < 200


def test_convergence_stops_at_max_iter() -> None:
    result = Study(_problem(OC(move=0.2)), max_iter=3, tol=1e-12).run()
    assert not result.converged
    assert result.iterations == 3
    assert "max" in result.reason.lower()


def test_x0_default_uses_volume_fraction() -> None:
    study = Study(_problem(), max_iter=1, tol=0.0)
    states = list(study.iterate())
    # first physical density averages near the volume target before optimization moves it
    assert 0.3 < states[0].x.mean() < 0.5  # started near vf=0.4


def test_explicit_x0_validated() -> None:
    p = _problem()
    with pytest.raises(ProblemError, match="x0"):
        Study(p, x0=np.ones(5))  # wrong size


def test_max_iter_must_be_positive() -> None:
    with pytest.raises(ProblemError, match="max_iter"):
        Study(_problem(), max_iter=0)


def test_multi_load_runs() -> None:
    g = StructuredGrid.box(size=(10.0, 10.0), shape=(10, 10))
    left = PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0), tol=1e-9)
    from topokit.selection import NearPoint

    model = LinearElasticity(
        g,
        Material(E=1.0, nu=0.3, rho=1.0),
        supports=[(left, "all")],
        loads=[
            [PointLoad(NearPoint((10.0, 10.0)), (0.0, -1.0))],
            [PointLoad(NearPoint((10.0, 0.0)), (0.0, -1.0))],
        ],
    )
    p = Problem(
        model,
        DensityFilter(radius=1.5) | SIMP(p=3.0),
        objective=Compliance(),
        constraints=[Volume() <= 0.4],
        optimizer=OC(move=0.2),
    )
    result = Study(p, max_iter=30, tol=1e-3).run()
    assert result.objective > 0.0
    assert result.history["volume"][-1] == pytest.approx(0.4, abs=1e-3)


def test_result_fields_and_history() -> None:
    result = Study(_problem(), max_iter=10, tol=0.0).run()
    assert result.design.values.shape == (_cantilever().mesh.n_elements,)
    assert len(result.history["objective"]) == result.iterations == 10
    assert result.timing >= 0.0


def test_study_is_deterministic() -> None:
    r1 = Study(_problem(MMA()), max_iter=15, tol=0.0).run()
    r2 = Study(_problem(MMA()), max_iter=15, tol=0.0).run()
    np.testing.assert_array_equal(r1.x, r2.x)


def test_amg_solver_runs() -> None:
    result = Study(_problem(solver=AmgCG(tol=1e-9)), max_iter=15, tol=1e-3).run()
    assert result.objective > 0.0
