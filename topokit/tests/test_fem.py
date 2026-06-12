"""Tier-1 tests for linear elasticity: elements, loads, patch and analytic cases."""

import numpy as np
import pytest
import scipy.sparse.linalg

from topokit.fem import (
    ALUMINUM_6061,
    STEEL,
    BodyForce,
    FemError,
    LinearElasticity,
    Material,
    PhysicsModel,
    PointLoad,
    SurfaceTraction,
)
from topokit.fields import FieldSpec
from topokit.mesh import StructuredGrid
from topokit.selection import Box, NearPoint, OnBoundary, PlaneSlab


def _solve(model: LinearElasticity, scale: np.ndarray) -> np.ndarray:
    from typing import Any, cast

    k = cast(Any, model.assemble(scale))  # .raw is the WP-1.1 interim accessor
    f = model.loads()
    u: np.ndarray = scipy.sparse.linalg.spsolve(k.raw.tocsc(), f[:, 0])
    return np.atleast_1d(u)


def _ones(mesh: StructuredGrid) -> np.ndarray:
    return np.ones(mesh.n_elements)


def test_material_validation() -> None:
    with pytest.raises(FemError, match="E"):
        Material(E=-1.0, nu=0.3, rho=1.0)
    with pytest.raises(FemError, match="nu"):
        Material(E=1.0, nu=0.5, rho=1.0)
    assert STEEL.E == 210000.0
    assert ALUMINUM_6061.nu == 0.33


def _cantilever(
    shape: tuple[int, ...],
    size: tuple[float, ...],
    material: Material,
    mode: str = "plane_stress",
) -> LinearElasticity:
    g = StructuredGrid.box(size=size, shape=shape)
    left = PlaneSlab(point=(0.0,) * g.dim, normal=(1.0,) + (0.0,) * (g.dim - 1), tol=1e-9)
    right_x = size[0]
    tip = Box((right_x, 0.0) + (0.0,) * (g.dim - 2), (right_x, *size[1:]), tol=1e-9)
    return LinearElasticity(
        g,
        material,
        supports=[(left, "all")],
        loads=[PointLoad(tip, force=(0.0, -1.0) + (0.0,) * (g.dim - 2))],
        mode=mode,
    )


def test_satisfies_physics_protocol() -> None:
    model: PhysicsModel = _cantilever((4, 2), (4.0, 2.0), STEEL)
    assert model.expected_field == FieldSpec("stiffness_scale")
    assert model.n_cases == 1


def test_element_stiffness_properties_2d() -> None:
    g = StructuredGrid(shape=(1, 1), spacing=(1.0, 1.0))
    m = LinearElasticity(
        g, STEEL, supports=[(NearPoint((0.0, 0.0)), "all")], loads=[BodyForce((0.0, -1.0))]
    )
    ke = m.element_stiffness
    assert ke.shape == (8, 8)
    np.testing.assert_allclose(ke, ke.T, atol=1e-9)
    eig = np.linalg.eigvalsh(ke)
    assert np.sum(np.abs(eig) < 1e-6 * eig.max()) == 3  # 2 translations + 1 rotation
    assert eig.min() > -1e-9 * eig.max()


def test_element_stiffness_properties_3d() -> None:
    g = StructuredGrid(shape=(1, 1, 1), spacing=(1.0, 1.0, 1.0))
    m = LinearElasticity(
        g,
        STEEL,
        supports=[(NearPoint((0.0, 0.0, 0.0)), "all")],
        loads=[BodyForce((0.0, 0.0, -1.0))],
    )
    ke = m.element_stiffness
    assert ke.shape == (24, 24)
    np.testing.assert_allclose(ke, ke.T, atol=1e-6)
    eig = np.linalg.eigvalsh(ke)
    assert np.sum(np.abs(eig) < 1e-6 * eig.max()) == 6  # 3 translations + 3 rotations


def test_2d_stiffness_invariant_under_uniform_scaling() -> None:
    a = LinearElasticity(
        StructuredGrid(shape=(1, 1), spacing=(1.0, 1.0)),
        STEEL,
        supports=[(NearPoint((0.0, 0.0)), "all")],
        loads=[BodyForce((0.0, -1.0))],
    ).element_stiffness
    b = LinearElasticity(
        StructuredGrid(shape=(1, 1), spacing=(3.0, 3.0)),
        STEEL,
        supports=[(NearPoint((0.0, 0.0)), "all")],
        loads=[BodyForce((0.0, -1.0))],
    ).element_stiffness
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_3d_stiffness_scales_linearly() -> None:
    def ke(h: float) -> np.ndarray:
        return LinearElasticity(
            StructuredGrid(shape=(1, 1, 1), spacing=(h, h, h)),
            STEEL,
            supports=[(NearPoint((0.0, 0.0, 0.0)), "all")],
            loads=[BodyForce((0.0, 0.0, -1.0))],
        ).element_stiffness

    np.testing.assert_allclose(ke(2.0), 2.0 * ke(1.0), rtol=1e-12)


def test_dof_condensation_and_supports() -> None:
    g = StructuredGrid(shape=(2, 1), spacing=(1.0, 1.0), void=[False, True])
    left = PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0), tol=1e-9)
    m = LinearElasticity(
        g, STEEL, supports=[(left, "all")], loads=[PointLoad(NearPoint((1.0, 0.5)), (1.0, 0.0))]
    )
    # 4 active nodes, 2 fully fixed -> 4 free dofs
    assert m.n_dof == 4


def test_support_selecting_nothing_raises() -> None:
    g = StructuredGrid(shape=(2, 2), spacing=(1.0, 1.0))
    far = Box((50.0, 50.0), (51.0, 51.0), tol=0.0)
    with pytest.raises(FemError, match="no active nodes"):
        LinearElasticity(g, STEEL, supports=[(far, "all")], loads=[BodyForce((0.0, -1.0))])


def test_traction_consistent_nodal_values() -> None:
    g = StructuredGrid(shape=(2, 2), spacing=(1.0, 1.0))
    left = PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0), tol=1e-9)
    right = Box((2.0, 0.0), (2.0, 2.0), tol=1e-9)
    m = LinearElasticity(
        g,
        STEEL,
        supports=[(left, "all")],
        loads=[SurfaceTraction(right, traction=(10.0, 0.0))],
    )
    f = m.loads()[:, 0]
    assert f[m.dof_index(5, 0)] == pytest.approx(10.0)  # mid-edge node, two faces
    assert f[m.dof_index(2, 0)] == pytest.approx(5.0)
    assert f[m.dof_index(8, 0)] == pytest.approx(5.0)
    assert f.sum() == pytest.approx(20.0)  # total = t * area


def test_pressure_acts_against_outward_normal() -> None:
    g = StructuredGrid(shape=(1, 1), spacing=(1.0, 1.0))
    left = PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0), tol=1e-9)
    top = Box((0.0, 1.0), (1.0, 1.0), tol=1e-9)
    m = LinearElasticity(
        g, STEEL, supports=[(left, "all")], loads=[SurfaceTraction(top, pressure=4.0)]
    )
    f = m.loads()[:, 0]
    assert f[m.dof_index(3, 1)] == pytest.approx(-2.0)  # pushes down on +y face
    assert f[m.dof_index(2, 1)] == pytest.approx(-2.0)


def test_point_load_split_equally() -> None:
    g = StructuredGrid(shape=(2, 2), spacing=(1.0, 1.0))
    pin = NearPoint((0.0, 0.0))
    right_edge = Box((2.0, 0.0), (2.0, 2.0), tol=1e-9)  # 3 nodes
    m = LinearElasticity(
        g, STEEL, supports=[(pin, "all")], loads=[PointLoad(right_edge, (0.0, -9.0))]
    )
    f = m.loads()[:, 0]
    for node in (2, 5, 8):
        assert f[m.dof_index(node, 1)] == pytest.approx(-3.0)


def test_body_force_consistent() -> None:
    g = StructuredGrid(shape=(2, 2), spacing=(1.0, 1.0))
    m = LinearElasticity(
        g, STEEL, supports=[(NearPoint((0.0, 0.0)), "all")], loads=[BodyForce((0.0, -1.0))]
    )
    f = m.loads()[:, 0]
    assert f[m.dof_index(4, 1)] == pytest.approx(-1.0)  # interior node, 4 elements
    assert f[m.dof_index(8, 1)] == pytest.approx(-0.25)  # corner
    assert f.sum() == pytest.approx(-4.0 + 0.25)  # total minus the pinned corner share


def test_multi_load_cases() -> None:
    g = StructuredGrid(shape=(2, 2), spacing=(1.0, 1.0))
    left = PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0), tol=1e-9)
    m = LinearElasticity(
        g,
        STEEL,
        supports=[(left, "all")],
        loads=[
            [PointLoad(NearPoint((2.0, 2.0)), (0.0, -1.0))],
            [PointLoad(NearPoint((2.0, 0.0)), (1.0, 0.0))],
        ],
    )
    assert m.n_cases == 2
    assert m.loads().shape == (m.n_dof, 2)


def test_energy_identity() -> None:
    m = _cantilever((8, 4), (8.0, 4.0), STEEL)
    scale = _ones(m.mesh)
    u = _solve(m, scale)
    k = m.assemble(scale)
    energy = float(u @ k.matvec(u))
    work = float(m.loads()[:, 0] @ u)
    assert energy == pytest.approx(work, rel=1e-10)
    assert m.element_energies(u, scale).sum() == pytest.approx(energy, rel=1e-9)


def test_uniaxial_patch_2d_exact() -> None:
    g = StructuredGrid(shape=(4, 2), spacing=(1.0, 1.0))
    e, sigma = 100.0, 10.0
    mat = Material(E=e, nu=0.3, rho=1.0)
    left = PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0), tol=1e-9)
    right = Box((4.0, 0.0), (4.0, 2.0), tol=1e-9)
    m = LinearElasticity(
        g,
        mat,
        supports=[(left, "x"), (NearPoint((0.0, 0.0)), "y")],
        loads=[SurfaceTraction(right, traction=(sigma, 0.0))],
    )
    scale = _ones(g)
    u = _solve(m, scale)
    disp = m.displacement_field(u)
    tip = disp.values[g.nodes[:, 0] == 4.0, 0]
    np.testing.assert_allclose(tip, sigma * 4.0 / e, rtol=1e-9)
    vm = m.element_stress(u, scale)
    np.testing.assert_allclose(vm, sigma, rtol=1e-9)


def test_uniaxial_patch_3d_exact() -> None:
    g = StructuredGrid(shape=(2, 1, 1), spacing=(1.0, 1.0, 1.0))
    e, sigma = 100.0, 10.0
    mat = Material(E=e, nu=0.3, rho=1.0)
    face_x0 = PlaneSlab(point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), tol=1e-9)
    face_x2 = Box((2.0, 0.0, 0.0), (2.0, 1.0, 1.0), tol=1e-9)
    supports = [
        (face_x0, "x"),
        (NearPoint((0.0, 0.0, 0.0)), ("y", "z")),
        (NearPoint((0.0, 0.0, 1.0)), ("y",)),
    ]
    m = LinearElasticity(
        g, mat, supports=supports, loads=[SurfaceTraction(face_x2, traction=(sigma, 0, 0))]
    )
    scale = _ones(g)
    u = _solve(m, scale)
    vm = m.element_stress(u, scale)
    np.testing.assert_allclose(vm, sigma, rtol=1e-9)
    disp = m.displacement_field(u)
    tip = disp.values[g.nodes[:, 0] == 2.0, 0]
    np.testing.assert_allclose(tip, sigma * 2.0 / e, rtol=1e-9)


def test_plane_strain_stiffening_ratio() -> None:
    nu = 0.3

    def tip_disp(mode: str) -> float:
        g = StructuredGrid(shape=(4, 2), spacing=(1.0, 1.0))
        mat = Material(E=100.0, nu=nu, rho=1.0)
        left = PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0), tol=1e-9)
        right = Box((4.0, 0.0), (4.0, 2.0), tol=1e-9)
        m = LinearElasticity(
            g,
            mat,
            supports=[(left, "x"), (NearPoint((0.0, 0.0)), "y")],
            loads=[SurfaceTraction(right, traction=(10.0, 0.0))],
            mode=mode,
        )
        u = _solve(m, _ones(g))
        return float(m.displacement_field(u).values[:, 0].max())

    ratio = tip_disp("plane_strain") / tip_disp("plane_stress")
    assert ratio == pytest.approx(1.0 - nu**2, rel=1e-9)


def test_cantilever_matches_timoshenko() -> None:
    e, nu, length, height, p = 1000.0, 0.3, 6.0, 1.0, 1.0
    inertia = height**3 / 12.0
    shear_g = e / (2.0 * (1.0 + nu))
    kappa = 5.0 / 6.0
    reference = p * length**3 / (3.0 * e * inertia) + p * length / (kappa * shear_g * height)

    def tip_deflection(shape: tuple[int, int]) -> float:
        m = _cantilever(shape, (length, height), Material(E=e, nu=nu, rho=1.0))
        u = _solve(m, np.ones(shape[0] * shape[1]))
        return float(-m.displacement_field(u).values[:, 1].min())

    coarse = tip_deflection((24, 4))
    fine = tip_deflection((48, 8))
    assert abs(fine - reference) / reference < 0.05
    assert abs(fine - reference) < abs(coarse - reference)  # converging


def test_void_elements_have_zero_energy_and_stress() -> None:
    void = np.zeros(8, dtype=bool)
    void[7] = True
    g = StructuredGrid(shape=(4, 2), spacing=(1.0, 1.0), void=void)
    left = PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0), tol=1e-9)
    m = LinearElasticity(
        g,
        STEEL,
        supports=[(left, "all")],
        loads=[PointLoad(NearPoint((4.0, 0.0)), (0.0, -1.0))],
    )
    scale = _ones(g)
    u = _solve(m, scale)
    assert m.element_energies(u, scale)[7] == 0.0
    assert m.element_stress(u, scale)[7] == 0.0
    assert m.element_energies(u, scale)[0] > 0.0


def test_scale_validation() -> None:
    m = _cantilever((4, 2), (4.0, 2.0), STEEL)
    with pytest.raises(FemError, match="shape"):
        m.assemble(np.ones(3))
    bad = np.ones(8)
    bad[0] = np.nan
    with pytest.raises(FemError, match="finite"):
        m.assemble(bad)


def test_traction_requires_exactly_one_of_traction_pressure() -> None:
    with pytest.raises(FemError, match="exactly one"):
        SurfaceTraction(OnBoundary(), traction=(1.0, 0.0), pressure=2.0)
    with pytest.raises(FemError, match="exactly one"):
        SurfaceTraction(OnBoundary())


def test_linear_elasticity_registered_as_builtin() -> None:
    from topokit.registry import registry

    assert registry.get("physics", "linear_elasticity") is LinearElasticity


def test_mode_rejected_for_3d() -> None:
    g = StructuredGrid(shape=(1, 1, 1), spacing=(1.0, 1.0, 1.0))
    with pytest.raises(FemError, match="2D"):
        LinearElasticity(
            g,
            STEEL,
            supports=[(NearPoint((0.0, 0.0, 0.0)), "all")],
            loads=[BodyForce((0.0, 0.0, -1.0))],
            mode="plane_strain",
        )


def test_negative_scale_rejected() -> None:
    m = _cantilever((4, 2), (4.0, 2.0), STEEL)
    bad = np.ones(8)
    bad[3] = -0.1
    with pytest.raises(FemError, match="negative"):
        m.assemble(bad)


def test_dof_index_component_validated() -> None:
    m = _cantilever((4, 2), (4.0, 2.0), STEEL)
    with pytest.raises(FemError, match="component"):
        m.dof_index(0, 2)


def test_empty_and_mixed_load_cases_rejected() -> None:
    g = StructuredGrid(shape=(2, 2), spacing=(1.0, 1.0))
    pin = NearPoint((0.0, 0.0))
    with pytest.raises(FemError, match="empty"):
        LinearElasticity(g, STEEL, supports=[(pin, "all")], loads=[[]])
    with pytest.raises(FemError, match="one case"):
        LinearElasticity(
            g,
            STEEL,
            supports=[(pin, "all")],
            loads=[BodyForce((0.0, -1.0)), [BodyForce((0.0, -1.0))]],  # type: ignore[arg-type]
        )


def test_load_params_coerce_to_canonical() -> None:
    sel = NearPoint((0.0, 0.0))
    assert PointLoad(sel, [0.0, -1.0]) == PointLoad(sel, (0, -1))  # type: ignore[arg-type]
    assert BodyForce([0.0, -1.0]) == BodyForce((0.0, -1.0))  # type: ignore[arg-type]


def test_model_arrays_read_only() -> None:
    m = _cantilever((4, 2), (4.0, 2.0), STEEL)
    with pytest.raises(ValueError, match="read-only"):
        m.element_stiffness[0, 0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        m.loads()[0, 0] = 0.0
