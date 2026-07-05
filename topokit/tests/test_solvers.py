# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Tests for linear solvers."""

from typing import Any

import numpy as np
import pytest

from topokit.backend import NumpyBackend, SparseMatrix
from topokit.fem import LinearElasticity, Material, PointLoad
from topokit.mesh import StructuredGrid
from topokit.selection import Box, NearPoint, PlaneSlab
from topokit.solvers import AmgCG, Direct, LinearSolver, SolverError, auto_solver

BK = NumpyBackend()


def _spd_system() -> tuple[SparseMatrix, np.ndarray, np.ndarray]:
    # 3x3 SPD matrix with known solution
    rows = BK.asarray([0, 0, 1, 1, 1, 2, 2], dtype=np.int64)
    cols = BK.asarray([0, 1, 0, 1, 2, 1, 2], dtype=np.int64)
    vals = BK.asarray([4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0])
    k = BK.coo_to_csr(rows, cols, vals, shape=(3, 3))
    x_ref = np.array([1.0, -2.0, 3.0])
    b = np.array([4.0 * 1 + 1.0 * (-2), 1.0 * 1 + 3.0 * (-2) + 1.0 * 3, 1.0 * (-2) + 2.0 * 3])
    return k, b, x_ref


def _cantilever_system(scale: np.ndarray | None = None) -> tuple[SparseMatrix, np.ndarray]:
    g = StructuredGrid.box(size=(12.0, 4.0), shape=(24, 8))
    left = PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0), tol=1e-9)
    tip = Box((12.0, 0.0), (12.0, 4.0), tol=1e-9)
    m = LinearElasticity(
        g,
        Material(E=1000.0, nu=0.3, rho=1.0),
        supports=[(left, "all")],
        loads=[PointLoad(tip, (0.0, -1.0))],
    )
    s = np.ones(g.n_elements) if scale is None else scale
    return m.assemble(s), m.loads()[:, 0]


def test_direct_solves_known_system() -> None:
    k, b, x_ref = _spd_system()
    s = Direct()
    s.prepare(k)
    np.testing.assert_allclose(s.solve(b), x_ref, rtol=1e-12)


def test_direct_multi_rhs() -> None:
    k, b, x_ref = _spd_system()
    s = Direct()
    s.prepare(k)
    out = s.solve(np.stack([b, 2.0 * b], axis=1))
    np.testing.assert_allclose(out[:, 0], x_ref, rtol=1e-12)
    np.testing.assert_allclose(out[:, 1], 2.0 * x_ref, rtol=1e-12)


def test_solve_before_prepare_raises() -> None:
    with pytest.raises(SolverError, match="prepare"):
        Direct().solve(np.ones(3))


def test_amg_cg_matches_direct_on_fem_system() -> None:
    k, f = _cantilever_system()
    d = Direct()
    d.prepare(k)
    u_ref = d.solve(f)
    a = AmgCG(tol=1e-10)
    a.prepare(k)
    u = a.solve(f)
    np.testing.assert_allclose(u, u_ref, rtol=1e-6, atol=1e-12 * np.abs(u_ref).max())


def test_amg_cg_respects_tolerance() -> None:
    k, f = _cantilever_system()
    a = AmgCG(tol=1e-8)
    a.prepare(k)
    u = a.solve(f)
    residual = float(np.linalg.norm(np.asarray(k.matvec(u)) - f))
    assert residual <= 1e-8 * float(np.linalg.norm(f)) * 10.0


def test_amg_cg_nonconvergence_raises() -> None:
    k, f = _cantilever_system()
    a = AmgCG(tol=1e-12, max_iter=1)
    a.prepare(k)
    with pytest.raises(SolverError, match="converge"):
        a.solve(f)


def test_amg_cg_iteration_count_resets_on_failed_solve() -> None:
    k, f = _cantilever_system()
    a = AmgCG(tol=1e-10)
    a.prepare(k)
    a.solve(f)
    assert a.last_iterations > 0  # a good solve recorded a count
    a.max_iter = 1  # force non-convergence on the next solve
    with pytest.raises(SolverError, match="converge"):
        a.solve(f)
    assert a.last_iterations == 0  # reset up front, not left stale from the prior solve


def test_amg_cg_missing_pyamg_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    import topokit.solvers as solvers_mod

    def boom() -> object:
        raise ImportError("no pyamg")

    monkeypatch.setattr(solvers_mod, "_load_pyamg", boom)
    k, _ = _spd_system()[0], None
    a = AmgCG()
    with pytest.raises(SolverError, match=r"topokit\[fast\]"):
        a.prepare(k)


def test_simp_regime_amg_vs_direct() -> None:
    rng = np.random.default_rng(seed=20260612)
    scale = np.where(rng.random(24 * 8) > 0.5, 1.0, 1e-9)
    scale[:8] = 1.0
    k, f = _cantilever_system(scale)
    d = Direct()
    d.prepare(k)
    u_ref = d.solve(f)
    a = AmgCG(tol=1e-10, max_iter=2000)
    a.prepare(k)
    u = a.solve(f)
    # accuracy is conditioning-limited in the SIMP regime (see WP-1.4 review)
    err = np.linalg.norm(u - u_ref) / np.linalg.norm(u_ref)
    assert err < 1e-4


def test_auto_solver_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    import topokit.solvers as solvers_mod

    assert isinstance(auto_solver(1000, dim=2), Direct)
    assert isinstance(auto_solver(100_000, dim=2), Direct)
    assert isinstance(auto_solver(500_000, dim=2), AmgCG)
    assert isinstance(auto_solver(1000, dim=3), Direct)
    assert isinstance(auto_solver(50_000, dim=3), AmgCG)
    with pytest.raises(SolverError, match="dim"):
        auto_solver(1000, dim=4)

    def boom() -> object:
        raise ImportError("no pyamg")

    monkeypatch.setattr(solvers_mod, "_load_pyamg", boom)
    with pytest.warns(UserWarning, match=r"topokit\[fast\]"):
        s = auto_solver(500_000, dim=3)
    assert isinstance(s, Direct)


def test_satisfies_protocol_and_registry() -> None:
    from topokit.registry import registry

    s: LinearSolver = Direct()
    assert s is not None
    assert registry.get("solvers", "direct") is Direct
    assert registry.get("solvers", "amg_cg") is AmgCG


def test_amg_cg_records_iteration_count() -> None:
    k, f = _cantilever_system()
    a = AmgCG(tol=1e-10)
    assert a.last_iterations == 0  # nothing solved yet
    a.prepare(k)
    a.solve(f)
    assert isinstance(a.last_iterations, int)
    assert 0 < a.last_iterations <= 2000


def test_amg_cg_iteration_count_is_worst_over_rhs() -> None:
    k, f = _cantilever_system()
    a = AmgCG(tol=1e-10)
    a.prepare(k)
    a.solve(np.stack([f, 2.0 * f], axis=1))
    assert a.last_iterations > 0


def test_amg_cg_multi_rhs() -> None:
    k, f = _cantilever_system()
    a = AmgCG(tol=1e-10)
    a.prepare(k)
    out = a.solve(np.stack([f, 2.0 * f], axis=1))
    np.testing.assert_allclose(out[:, 1], 2.0 * out[:, 0], rtol=1e-8)


def _singular_system() -> tuple[SparseMatrix, np.ndarray]:
    from topokit.fem import STEEL
    from topokit.selection import NearPoint

    g = StructuredGrid(shape=(2, 2), spacing=(1.0, 1.0))
    m = LinearElasticity(
        g,
        STEEL,
        supports=[(NearPoint((0.0, 0.0)), "x")],  # rigid modes remain
        loads=[PointLoad(Box((2.0, 0.0), (2.0, 2.0), tol=1e-9), (0.0, -1.0))],
    )
    return m.assemble(np.ones(4)), m.loads()[:, 0]


def test_direct_detects_singular_system() -> None:
    k, f = _singular_system()
    d = Direct()
    d.prepare(k)
    with pytest.raises(SolverError, match="singular"):
        d.solve(f)


def test_direct_residual_check_can_be_disabled() -> None:
    k, f = _singular_system()
    d = Direct(residual_check=None)
    d.prepare(k)
    d.solve(f)  # garbage, but explicitly requested


def test_direct_prefers_cholmod_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    import scipy.sparse.linalg as sla

    calls: list[str] = []

    def fake_cholesky(csc: Any) -> object:
        calls.append("cholmod")
        return sla.splu(csc).solve

    chol = types.ModuleType("sksparse.cholmod")
    chol.cholesky = fake_cholesky  # type: ignore[attr-defined]
    sk = types.ModuleType("sksparse")
    sk.cholmod = chol  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sksparse", sk)
    monkeypatch.setitem(sys.modules, "sksparse.cholmod", chol)
    k, b, x_ref = _spd_system()
    d = Direct()
    d.prepare(k)
    assert calls == ["cholmod"]
    np.testing.assert_allclose(d.solve(b), x_ref, rtol=1e-12)


def test_solver_param_validation() -> None:
    with pytest.raises(SolverError, match="tol"):
        AmgCG(tol=0.0)
    with pytest.raises(SolverError, match="max_iter"):
        AmgCG(max_iter=0)
    with pytest.raises(SolverError, match="residual_check"):
        Direct(residual_check=-1.0)


def test_amg_cg_is_deterministic() -> None:
    k, f = _cantilever_system()
    a1 = AmgCG(tol=1e-10)
    a1.prepare(k)
    u1 = a1.solve(f)
    a2 = AmgCG(tol=1e-10)
    a2.prepare(k)
    u2 = a2.solve(f)
    assert np.array_equal(u1, u2)  # bitwise, decision E11


def test_amg_cg_does_not_disturb_global_rng() -> None:
    k, _ = _cantilever_system()
    np.random.seed(123)
    expected = np.random.RandomState(123).random(3)
    a = AmgCG()
    a.prepare(k)
    np.testing.assert_array_equal(np.random.random(3), expected)


def test_direct_wraps_cholmod_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # a numerical CHOLMOD failure must surface as SolverError, like the splu path
    import sys
    import types

    def failing_cholesky(csc: Any) -> object:
        raise RuntimeError("matrix not positive definite")

    chol = types.ModuleType("sksparse.cholmod")
    chol.cholesky = failing_cholesky  # type: ignore[attr-defined]
    sk = types.ModuleType("sksparse")
    sk.cholmod = chol  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sksparse", sk)
    monkeypatch.setitem(sys.modules, "sksparse.cholmod", chol)
    k, _b, _x = _spd_system()
    with pytest.raises(SolverError, match="factorization failed"):
        Direct().prepare(k)


def _cantilever_3d_model(n: int) -> LinearElasticity:
    g = StructuredGrid.box(size=(float(2 * n), float(n), float(n)), shape=(2 * n, n, n))
    left = PlaneSlab(point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), tol=1e-9)
    return LinearElasticity(
        g,
        Material(E=1.0, nu=0.3, rho=1.0),
        supports=[(left, "all")],
        loads=[PointLoad(NearPoint((float(2 * n), n / 2.0, n / 2.0)), (0.0, -1.0, 0.0))],
    )


def test_near_nullspace_cuts_cg_iterations() -> None:
    pytest.importorskip("pyamg")
    model = _cantilever_3d_model(10)  # 20x10x10, ~7k free DOFs
    k = model.assemble(np.full(model.mesh.n_elements, 0.5))
    base = AmgCG()
    base.prepare(k)
    base.solve(model.loads())
    rbm = AmgCG()
    rbm.set_near_nullspace(model.near_nullspace())
    rbm.prepare(k)
    u = rbm.solve(model.loads())
    assert rbm.last_iterations <= 0.75 * base.last_iterations  # conservative; ~2-3x expected
    # and the answer is still the answer (atol above the ~1e-8-tol CG noise floor,
    # not 1e-12: two independently-converged solves agree elementwise only down
    # to their own residual tolerance, and ~9% of free DOFs here are small enough
    # for that noise to dominate a per-element check at 1e-12)
    np.testing.assert_allclose(
        np.asarray(u), np.asarray(base.solve(model.loads())), rtol=1e-6, atol=1e-6
    )


def test_set_near_nullspace_validation() -> None:
    s = AmgCG()
    with pytest.raises(SolverError, match="modes"):
        s.set_near_nullspace(np.ones(5))  # 1-D
    pytest.importorskip("pyamg")
    model = _cantilever_3d_model(2)
    k = model.assemble(np.full(model.mesh.n_elements, 0.5))
    s2 = AmgCG()
    s2.set_near_nullspace(np.ones((3, 6)))  # wrong row count
    with pytest.raises(SolverError, match="rows"):
        s2.prepare(k)


def test_failed_prepare_leaves_solver_unprepared() -> None:
    pytest.importorskip("pyamg")
    model = _cantilever_3d_model(2)
    k = model.assemble(np.full(model.mesh.n_elements, 0.5))
    s = AmgCG()
    s.set_near_nullspace(np.ones((3, 6)))  # wrong row count
    with pytest.raises(SolverError, match="rows"):
        s.prepare(k)
    with pytest.raises(SolverError, match="prepare"):
        s.solve(model.loads())


def test_near_nullspace_deterministic_hierarchy() -> None:
    pytest.importorskip("pyamg")
    model = _cantilever_3d_model(4)
    k = model.assemble(np.full(model.mesh.n_elements, 0.5))
    us = []
    for _ in range(2):
        s = AmgCG()
        s.set_near_nullspace(model.near_nullspace())
        s.prepare(k)
        us.append(np.asarray(s.solve(model.loads())))
    np.testing.assert_array_equal(us[0], us[1])
