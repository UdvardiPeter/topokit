"""Tests for the Problem/Study orchestration layer."""

from pathlib import Path

import numpy as np
import pytest

from topokit.checkpoint import read_topo
from topokit.events import (
    FieldSnapshot,
    IterationFinished,
    StageFinished,
    StudyFinished,
    StudyStarted,
)
from topokit.fem import LinearElasticity, Material, PointLoad
from topokit.fields import FieldSpec
from topokit.mesh import StructuredGrid
from topokit.optimizers import MMA, OC, Optimizer
from topokit.parametrization import SIMP, DensityFilter, Heaviside, SymmetryMap
from topokit.problem import Problem, ProblemError, Schedule, Study
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


def _problem_proj(optimizer: Optimizer | None = None) -> Problem:
    model = _cantilever()
    chain = DensityFilter(radius=1.5) | Heaviside(beta=1.0) | SIMP(p=3.0)
    return Problem(
        model,
        chain,
        objective=Compliance(),
        constraints=[Volume() <= 0.4],
        optimizer=optimizer or OC(move=0.2),
    )


def test_continuation_runs_stages_and_emits_stage_events() -> None:
    p = _problem_proj()
    stages_seen: list[StageFinished] = []
    study = Study(p, schedule=Schedule.default(max_iter=15, tol=1e-3))
    study.events.subscribe(StageFinished, stages_seen.append)
    result = study.run()
    assert len(stages_seen) == 8  # all stages executed (Heaviside present)
    assert result.history["stage"][-1] == 7  # zero-based last stage index
    assert result.history["stage"][0] == 0
    assert result.iterations == len(result.history["objective"])


def test_default_study_runs_continuation() -> None:
    # schedule=None -> Schedule.default -> continuation ON (E7)
    study = Study(_problem_proj(), max_iter=12, tol=1e-3)  # no schedule arg
    result = study.run()
    assert result.history["stage"][-1] >= 1  # more than one stage ran


def test_continuation_dedups_when_no_heaviside() -> None:
    # DensityFilter|SIMP: beta-ramp stages collapse to the distinct p stages (1,2,3)
    p = _problem(OC(move=0.2))
    stages_seen: list[StageFinished] = []
    study = Study(p, schedule=Schedule.default(max_iter=10, tol=1e-3))
    study.events.subscribe(StageFinished, stages_seen.append)
    study.run()
    assert len(stages_seen) == 3


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
    result = Study(p, schedule=Schedule.single(p=3.0, max_iter=5, tol=0.0)).run()
    assert len(result.history["vol_design"]) == 5
    assert len(result.history["vol_all"]) == 5


def test_study_reduces_compliance_with_oc() -> None:
    # OC plateaus on this grey (no-projection) cantilever rather than reaching a
    # tight change tol, so assert the optimization worked, not convergence.
    result = Study(
        _problem(OC(move=0.2)), schedule=Schedule.single(p=3.0, max_iter=120, tol=1e-3)
    ).run()
    assert result.history["objective"][-1] < 0.5 * result.history["objective"][0]
    assert result.history["volume"][-1] == pytest.approx(0.4, abs=1e-3)


def test_study_with_mma_matches_oc() -> None:
    # MMA needs objective normalization (Study supplies it); without it MMA
    # would converge wrong. Assert MMA reaches the same compliance as OC.
    c_oc = (
        Study(_problem(OC(move=0.2)), schedule=Schedule.single(p=3.0, max_iter=120, tol=1e-3))
        .run()
        .objective
    )
    c_mma = (
        Study(_problem(MMA()), schedule=Schedule.single(p=3.0, max_iter=120, tol=1e-3))
        .run()
        .objective
    )
    assert abs(c_mma - c_oc) / c_oc < 0.05


def test_study_emits_events() -> None:
    study = Study(
        _problem(), schedule=Schedule.single(p=3.0, max_iter=20, tol=0.0), snapshot_every=5
    )
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
    states = list(
        Study(_problem(), schedule=Schedule.single(p=3.0, max_iter=15, tol=0.0)).iterate()
    )
    assert len(states) == 15
    final_iter = Study(_problem(), schedule=Schedule.single(p=3.0, max_iter=15, tol=0.0)).run()
    np.testing.assert_allclose(states[-1].x, final_iter.x)


def test_convergence_stops_at_tol() -> None:
    # MMA reaches a tight change tol on the cantilever (~78 iters); OC plateaus
    result = Study(_problem(MMA()), schedule=Schedule.single(p=3.0, max_iter=200, tol=1e-3)).run()
    assert result.converged
    assert result.iterations < 200


def test_convergence_stops_at_max_iter() -> None:
    result = Study(
        _problem(OC(move=0.2)), schedule=Schedule.single(p=3.0, max_iter=3, tol=1e-12)
    ).run()
    assert not result.converged
    assert result.iterations == 3
    assert "max" in result.reason.lower()


def test_x0_default_uses_volume_fraction() -> None:
    study = Study(_problem(), schedule=Schedule.single(p=3.0, max_iter=1, tol=0.0))
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
    result = Study(p, schedule=Schedule.single(p=3.0, max_iter=30, tol=1e-3)).run()
    assert result.objective > 0.0
    assert result.history["volume"][-1] == pytest.approx(0.4, abs=1e-3)


def test_result_tracks_best_feasible() -> None:
    result = Study(
        _problem(OC(move=0.2)), schedule=Schedule.single(p=3.0, max_iter=40, tol=1e-3)
    ).run()
    assert result.best_objective <= result.objective + 1e-9
    assert result.best_x.shape == result.x.shape
    assert result.best_design.values.shape == (_cantilever().mesh.n_elements,)


def test_study_writes_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "run.topo"
    Study(
        _problem(OC(move=0.2)),
        schedule=Schedule.single(p=3.0, max_iter=12, tol=0.0),
        checkpoint_path=str(path),
        checkpoint_every=5,
    ).run()
    assert path.exists()
    m, a = read_topo(str(path))
    assert "x" in a and "fingerprint" in m and "c0" in m


def test_result_fields_and_history() -> None:
    result = Study(_problem(), schedule=Schedule.single(p=3.0, max_iter=10, tol=0.0)).run()
    assert result.design.values.shape == (_cantilever().mesh.n_elements,)
    assert len(result.history["objective"]) == result.iterations == 10
    assert result.timing >= 0.0


def test_study_is_deterministic() -> None:
    r1 = Study(_problem(MMA()), schedule=Schedule.single(p=3.0, max_iter=15, tol=0.0)).run()
    r2 = Study(_problem(MMA()), schedule=Schedule.single(p=3.0, max_iter=15, tol=0.0)).run()
    np.testing.assert_array_equal(r1.x, r2.x)


def test_resume_matches_uninterrupted(tmp_path: Path) -> None:
    full = Study(_problem(MMA()), schedule=Schedule.single(p=3.0, max_iter=30, tol=0.0)).run()

    path = tmp_path / "run.topo"
    Study(
        _problem(MMA()),
        schedule=Schedule.single(p=3.0, max_iter=12, tol=0.0),
        checkpoint_path=str(path),
        checkpoint_every=12,
    ).run()
    resumed = Study.resume(_problem(MMA()), str(path))
    resumed.schedule = Schedule.single(p=3.0, max_iter=30, tol=0.0)  # extend the cap
    out = resumed.run()
    np.testing.assert_array_equal(out.x, full.x)


def test_estimate_stiffness_bytes() -> None:
    from topokit.problem import _estimate_stiffness_bytes

    assert _estimate_stiffness_bytes(n_dof=1_000_000, nnz=80_000_000) == (
        80_000_000 * 12 + 1_000_001 * 4
    )


def test_small_problem_does_not_warn() -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any memory warning would fail the test
        _problem(OC(move=0.2))  # the 20x10 cantilever is tiny


def test_resume_completed_run_raises(tmp_path: Path) -> None:
    # resuming a checkpoint that is already at the end of its schedule, with the
    # same schedule, has nothing to do -> clear error rather than a crash
    path = tmp_path / "run.topo"
    Study(
        _problem(OC(move=0.2)),
        schedule=Schedule.single(p=3.0, max_iter=8, tol=0.0),
        checkpoint_path=str(path),
        checkpoint_every=8,
    ).run()
    resumed = Study.resume(_problem(OC(move=0.2)), str(path))  # same (finished) schedule
    with pytest.raises(ProblemError, match="nothing to resume"):
        resumed.run()


def test_resume_rejects_changed_optimizer_params(tmp_path: Path) -> None:
    path = tmp_path / "run.topo"
    Study(
        _problem(OC(move=0.2)),
        schedule=Schedule.single(p=3.0, max_iter=5, tol=0.0),
        checkpoint_path=str(path),
        checkpoint_every=5,
    ).run()
    # same chain/objective, but a different optimizer hyperparameter
    other = _problem(OC(move=0.5))
    with pytest.raises(ProblemError, match="different problem"):
        Study.resume(other, str(path))


def test_continuation_sharpens_vs_single_stage() -> None:
    # the point of continuation: ramping Heaviside beta drives the design toward
    # 0/1, so the grey fraction is lower than a fixed-beta single stage.
    def grey(rho: np.ndarray) -> float:
        return float(np.mean((rho > 0.1) & (rho < 0.9)))

    cont = Study(_problem_proj(), schedule=Schedule.default(max_iter=20, tol=1e-3)).run()
    single = Study(
        _problem_proj(), schedule=Schedule.single(p=3.0, beta=1.0, max_iter=20, tol=1e-3)
    ).run()
    assert grey(cont.design.values) < grey(single.design.values)


def test_resume_rejects_wrong_problem(tmp_path: Path) -> None:
    path = tmp_path / "run.topo"
    Study(
        _problem(OC(move=0.2)),
        schedule=Schedule.single(p=3.0, max_iter=5, tol=0.0),
        checkpoint_path=str(path),
        checkpoint_every=5,
    ).run()
    other = _problem_proj(OC(move=0.2))  # different chain (has Heaviside)
    with pytest.raises(ProblemError, match="different problem"):
        Study.resume(other, str(path))


def test_amg_solver_runs() -> None:
    result = Study(
        _problem(solver=AmgCG(tol=1e-9)), schedule=Schedule.single(p=3.0, max_iter=15, tol=1e-3)
    ).run()
    assert result.objective > 0.0


def test_symmetry_runs_in_reduced_space() -> None:
    # SymmetryMap is a reduced-input link: the optimizer works in the reduced
    # space, so n_vars < design-element count. Confirms the reduced gradient
    # size flows through optimizer.setup/step and the design vars stay reduced.
    model = _cantilever()
    chain = SymmetryMap(planes=("x",)) | DensityFilter(radius=1.5) | SIMP(p=3.0)
    p = Problem(
        model,
        chain,
        objective=Compliance(),
        constraints=[Volume() <= 0.4],
        optimizer=OC(move=0.2),
    )
    assert p.chain.n_vars < int(model.mesh.design.sum())  # space genuinely reduced
    result = Study(p, schedule=Schedule.single(p=3.0, max_iter=30, tol=1e-3)).run()
    assert result.x.size == p.chain.n_vars
    assert result.history["objective"][-1] < result.history["objective"][0]
    assert result.history["volume"][-1] == pytest.approx(0.4, abs=1e-3)


def test_staged_chain_overrides_p_and_beta() -> None:
    from topokit.problem import _staged_chain

    model = _cantilever()
    spec = DensityFilter(radius=1.5) | Heaviside(beta=1.0) | SIMP(p=3.0)
    staged = _staged_chain(spec, model.mesh, p=2.0, beta=8.0)
    simp = next(link for link in staged.spec.links if isinstance(link, SIMP))
    heav = next(link for link in staged.spec.links if isinstance(link, Heaviside))
    assert simp.p == 2.0
    assert heav.beta == 8.0


def test_staged_chains_equal_when_params_unchanged() -> None:
    # a chain without Heaviside: beta changes produce identical specs (dedup)
    from topokit.problem import _staged_chain

    model = _cantilever()
    spec = DensityFilter(radius=1.5) | SIMP(p=3.0)
    a = _staged_chain(spec, model.mesh, p=3.0, beta=1.0)
    b = _staged_chain(spec, model.mesh, p=3.0, beta=32.0)
    assert a.spec == b.spec  # frozen-dataclass equality -> dedup


def test_study_reports_kkt() -> None:
    study = Study(_problem(), schedule=Schedule.single(p=3.0, max_iter=10, tol=0.0))
    iters: list[IterationFinished] = []
    study.events.subscribe(IterationFinished, iters.append)
    result = study.run()
    assert all(np.isfinite(it.kkt) for it in iters)
    assert np.isfinite(result.kkt)
    assert len(result.history["kkt"]) == result.iterations
