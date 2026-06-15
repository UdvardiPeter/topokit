"""Tests for objective/constraint responses."""

from typing import Any

import numpy as np
import pytest

from topokit.fem import STEEL, LinearElasticity, Material, PointLoad
from topokit.mesh import StructuredGrid
from topokit.parametrization import SIMP, DensityFilter
from topokit.responses import (
    Compliance,
    Constraint,
    ResponseError,
    Solution,
    Volume,
    von_mises,
)
from topokit.selection import Box, NearPoint, PlaneSlab
from topokit.solvers import Direct
from topokit.testing import assert_gradient_matches


def _cantilever(
    shape: tuple[int, ...] = (12, 4), size: tuple[float, ...] = (12.0, 4.0), **mesh_kw: Any
) -> tuple[StructuredGrid, LinearElasticity]:
    g = StructuredGrid.box(size=size, shape=shape, **mesh_kw)
    left = PlaneSlab(point=(0.0,) * g.dim, normal=(1.0,) + (0.0,) * (g.dim - 1), tol=1e-9)
    tip = Box((size[0], 0.0), (size[0], size[1]), tol=1e-9)
    model = LinearElasticity(
        g,
        Material(E=1000.0, nu=0.3, rho=1.0),
        supports=[(left, "all")],
        loads=[PointLoad(tip, (0.0, -1.0))],
    )
    return g, model


def _solution(model: LinearElasticity, g: StructuredGrid, chain: Any, x: np.ndarray) -> Solution:
    scale = chain.apply(x)
    rho = chain.physical_density(x)
    solver = Direct()
    solver.prepare(model.assemble(scale))
    u = np.atleast_2d(np.asarray(solver.solve(model.loads()))).reshape(model.n_dof, -1)
    return Solution(model=model, mesh=g, displacements=u, interpolated=scale, density=rho)


def test_response_protocol_metadata() -> None:
    assert Compliance.field_basis == "interpolated"
    assert Volume.field_basis == "density"
    assert Compliance.n_extra_adjoints == 0
    assert Volume.n_extra_adjoints == 0


def test_compliance_value_equals_load_work() -> None:
    g, model = _cantilever()
    chain = SIMP(p=1.0, scale_min=1e-6).bind(g)
    x = np.full(chain.n_vars, 1.0)
    sol = _solution(model, g, chain, x)
    c = Compliance().value(sol)
    fu = float(model.loads()[:, 0] @ sol.displacements[:, 0])
    assert c == pytest.approx(fu, rel=1e-9)
    assert c > 0.0


def test_compliance_gradient_fd_through_full_stack() -> None:
    g, model = _cantilever()
    chain = (DensityFilter(radius=1.5) | SIMP(p=3.0)).bind(g)
    rng = np.random.default_rng(20260615)
    x = rng.uniform(0.3, 0.9, size=chain.n_vars)
    comp = Compliance()

    def f(xx: np.ndarray) -> float:
        return comp.value(_solution(model, g, chain, xx))

    def grad(xx: np.ndarray) -> np.ndarray:
        sol = _solution(model, g, chain, xx)
        return np.asarray(chain.pullback(xx, comp.grad_field(sol)))

    assert_gradient_matches(f, grad, x, rtol=1e-5)


def test_compliance_multi_load_weighted() -> None:
    g = StructuredGrid.box(size=(4.0, 4.0), shape=(4, 4))
    left = PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0), tol=1e-9)
    model = LinearElasticity(
        g,
        STEEL,
        supports=[(left, "all")],
        loads=[
            [PointLoad(NearPoint((4.0, 4.0)), (0.0, -1.0))],
            [PointLoad(NearPoint((4.0, 0.0)), (1.0, 0.0))],
        ],
    )
    chain = SIMP(p=1.0, scale_min=1e-6).bind(g)
    sol = _solution(model, g, chain, np.full(chain.n_vars, 1.0))
    c_each = [
        Compliance(weights=(1.0, 0.0)).value(sol),
        Compliance(weights=(0.0, 1.0)).value(sol),
    ]
    c_sum = Compliance().value(sol)  # default equal weights of 1.0
    assert c_sum == pytest.approx(c_each[0] + c_each[1], rel=1e-9)
    c_weighted = Compliance(weights=(2.0, 3.0)).value(sol)
    assert c_weighted == pytest.approx(2.0 * c_each[0] + 3.0 * c_each[1], rel=1e-9)


def test_compliance_weight_count_validated() -> None:
    g, model = _cantilever(shape=(4, 2), size=(4.0, 2.0))
    chain = SIMP().bind(g)
    sol = _solution(model, g, chain, np.full(chain.n_vars, 0.5))
    with pytest.raises(ResponseError, match="weights"):
        Compliance(weights=(1.0, 1.0)).value(sol)  # single load case


def test_volume_value_and_region() -> None:
    solid = np.zeros(8, dtype=bool)
    solid[0] = True
    void = np.zeros(8, dtype=bool)
    void[7] = True
    g = StructuredGrid(shape=(4, 2), spacing=(1.0, 1.0), solid=solid, void=void)
    chain = SIMP().bind(g)
    x = np.full(chain.n_vars, 0.5)
    sol = _solution_no_solve(g, chain, x)
    # design region: 6 design elements at rho=0.5 -> fraction 0.5
    assert Volume(region="design").value(sol) == pytest.approx(0.5)
    # all region: (6*0.5 + 1 solid) / 8 = 4/8 = 0.5; void contributes 0
    assert Volume(region="all").value(sol) == pytest.approx((6 * 0.5 + 1.0) / 8.0)


def _solution_no_solve(g: StructuredGrid, chain: Any, x: np.ndarray) -> Solution:
    scale = chain.apply(x)
    rho = chain.physical_density(x)
    return Solution(
        model=None, mesh=g, displacements=np.zeros((1, 1)), interpolated=scale, density=rho
    )


def test_volume_gradient_fd() -> None:
    g, _model = _cantilever(shape=(6, 4), size=(6.0, 4.0))
    chain = (DensityFilter(radius=1.5) | SIMP(p=3.0)).bind(g)
    rng = np.random.default_rng(7)
    x = rng.uniform(0.3, 0.9, size=chain.n_vars)
    vol = Volume()

    def f(xx: np.ndarray) -> float:
        return vol.value(_solution_no_solve(g, chain, xx))

    def grad(xx: np.ndarray) -> np.ndarray:
        sol = _solution_no_solve(g, chain, xx)
        return np.asarray(chain.pullback_density(xx, vol.grad_field(sol)))

    assert_gradient_matches(f, grad, x, rtol=1e-5)


def test_volume_gradient_is_constant_fraction() -> None:
    g = StructuredGrid(shape=(4, 2), spacing=(1.0, 2.0))  # v_e = 2.0
    chain = SIMP().bind(g)
    sol = _solution_no_solve(g, chain, np.full(chain.n_vars, 0.5))
    grad = Volume().grad_field(sol)
    # dV/d rho_e = v_e / sum(v_design) = 2 / 16 = 0.125, equal for all design
    np.testing.assert_allclose(grad, 1.0 / 8.0)


def test_constraint_normalization() -> None:
    g, _model = _cantilever(shape=(4, 2), size=(4.0, 2.0))
    chain = SIMP().bind(g)
    sol = _solution_no_solve(g, chain, np.full(chain.n_vars, 0.6))
    c = Volume() <= 0.3
    assert isinstance(c, Constraint)
    # V = 0.6, bound 0.3 -> g = 0.6/0.3 - 1 = 1.0
    assert c.value(sol) == pytest.approx(1.0)
    g_field = c.grad_field(sol)
    np.testing.assert_allclose(g_field, Volume().grad_field(sol) / 0.3)
    ge = Volume() >= 0.3
    assert ge.value(sol) == pytest.approx(1.0 - 0.6 / 0.3)


def test_von_mises_field() -> None:
    void = np.zeros(8, dtype=bool)
    void[7] = True
    g = StructuredGrid(shape=(4, 2), spacing=(1.0, 1.0), void=void)
    left = PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0), tol=1e-9)
    model = LinearElasticity(
        g, STEEL, supports=[(left, "all")], loads=[PointLoad(NearPoint((4.0, 0.0)), (0.0, -1.0))]
    )
    chain = SIMP().bind(g)
    sol = _solution(model, g, chain, np.full(chain.n_vars, 1.0))
    field = von_mises(sol)
    assert field.values.shape == (8,)
    assert field.values[7] == 0.0  # void
    assert field.name == "von_mises"


def test_responses_registry_fd_meta() -> None:
    from topokit.registry import registry

    g, model = _cantilever(shape=(6, 4), size=(6.0, 4.0))
    chain = (DensityFilter(radius=1.5) | SIMP(p=3.0)).bind(g)
    rng = np.random.default_rng(3)
    x = rng.uniform(0.3, 0.9, size=chain.n_vars)
    for name in registry.names("responses"):
        cls = registry.get("responses", name)
        if getattr(cls, "fd_exempt", None) is not None:
            continue
        resp = cls.fd_example()
        pull = chain.pullback if cls.field_basis == "interpolated" else chain.pullback_density

        def f(xx: np.ndarray, r: Any = resp) -> float:
            return float(r.value(_solution(model, g, chain, xx)))

        def grad(xx: np.ndarray, r: Any = resp, p: Any = pull) -> np.ndarray:
            return np.asarray(p(xx, r.grad_field(_solution(model, g, chain, xx))))

        assert_gradient_matches(f, grad, x, rtol=1e-5)
