# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Tier-1 tests for linear elasticity: elements, loads, patch and analytic cases."""

import numpy as np
import pytest
import scipy.sparse

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
    from topokit.solvers import Direct

    solver = Direct()
    solver.prepare(model.assemble(scale))
    return np.asarray(solver.solve(model.loads()[:, 0]))


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


def test_uniaxial_patch_anisotropic_offset_grid_exact() -> None:
    g = StructuredGrid(shape=(4, 2), spacing=(0.7, 0.31), origin=(5.0, -2.0))
    mat = Material(E=100.0, nu=0.3, rho=1.0)
    length = 4 * 0.7
    left = PlaneSlab(point=(5.0, 0.0), normal=(1.0, 0.0), tol=1e-9)
    right = Box((5.0 + length, -2.0), (5.0 + length, -2.0 + 0.62), tol=1e-9)
    m = LinearElasticity(
        g,
        mat,
        supports=[(left, "x"), (NearPoint((5.0, -2.0)), "y")],
        loads=[SurfaceTraction(right, traction=(10.0, 0.0))],
    )
    u = _solve(m, _ones(g))
    np.testing.assert_allclose(m.element_stress(u, _ones(g)), 10.0, rtol=1e-9)


def test_uniaxial_patch_with_solid_region_exact() -> None:
    solid = np.zeros(8, dtype=bool)
    solid[[1, 5]] = True  # one full column of solid elements
    g = StructuredGrid(shape=(4, 2), spacing=(1.0, 1.0), solid=solid)
    mat = Material(E=100.0, nu=0.3, rho=1.0)
    left = PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0), tol=1e-9)
    right = Box((4.0, 0.0), (4.0, 2.0), tol=1e-9)
    m = LinearElasticity(
        g,
        mat,
        supports=[(left, "x"), (NearPoint((0.0, 0.0)), "y")],
        loads=[SurfaceTraction(right, traction=(10.0, 0.0))],
    )
    u = _solve(m, _ones(g))
    np.testing.assert_allclose(m.element_stress(u, _ones(g)), 10.0, rtol=1e-9)


def test_plane_strain_von_mises_includes_sigma_z() -> None:
    g = StructuredGrid(shape=(4, 2), spacing=(1.0, 1.0))
    mat = Material(E=100.0, nu=0.3, rho=1.0)
    left = PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0), tol=1e-9)
    right = Box((4.0, 0.0), (4.0, 2.0), tol=1e-9)
    m = LinearElasticity(
        g,
        mat,
        supports=[(left, "x"), (NearPoint((0.0, 0.0)), "y")],
        loads=[SurfaceTraction(right, traction=(10.0, 0.0))],
        mode="plane_strain",
    )
    u = _solve(m, _ones(g))
    # sigma_z = nu*(sx+sy) = 3, vm = sqrt(0.5*((10)^2 + 3^2 + 7^2)) = sqrt(79)
    np.testing.assert_allclose(m.element_stress(u, _ones(g)), np.sqrt(79.0), rtol=1e-9)


def test_energy_consistency_in_simp_regime() -> None:
    m = _cantilever((12, 4), (12.0, 4.0), STEEL)
    rng = np.random.default_rng(seed=20260612)
    scale = np.where(rng.random(48) > 0.5, 1.0, 1e-9)
    scale[:4] = 1.0  # keep the clamped end stiff
    k = m.assemble(scale)
    # assembly/energy consistency holds for any u, independent of conditioning
    u_rand = rng.normal(size=m.n_dof)
    np.testing.assert_allclose(
        m.element_energies(u_rand, scale).sum(), u_rand @ k.matvec(u_rand), rtol=1e-9
    )
    # for the solved u the identity is limited by the solver residual:
    # cond(K) ~ 1e9 at this scale contrast, measured mismatch ~5e-6
    u = _solve(m, scale)
    energy = float(u @ k.matvec(u))
    work = float(m.loads()[:, 0] @ u)
    assert energy == pytest.approx(work, rel=1e-4)


def test_pressure_on_void_interface_face() -> None:
    g = StructuredGrid(shape=(2, 1), spacing=(1.0, 1.0), void=[False, True])
    interface = Box((1.0, 0.0), (1.0, 1.0), tol=1e-9)
    m = LinearElasticity(
        g,
        STEEL,
        supports=[(PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0), tol=1e-9), "all")],
        loads=[SurfaceTraction(interface, pressure=4.0)],
    )
    f = m.loads()[:, 0]
    # pressure pushes against the +x outward normal of the interface
    assert f[m.dof_index(1, 0)] == pytest.approx(-2.0)
    assert f[m.dof_index(4, 0)] == pytest.approx(-2.0)


def test_cantilever_3d_matches_timoshenko() -> None:
    e, nu, length, side, p = 1000.0, 0.3, 6.0, 1.0, 1.0
    inertia = side**4 / 12.0
    shear_g = e / (2.0 * (1.0 + nu))
    reference = p * length**3 / (3.0 * e * inertia) + p * length / (5.0 / 6.0 * shear_g * side**2)

    def tip(shape: tuple[int, int, int]) -> float:
        g = StructuredGrid.box(size=(length, side, side), shape=shape)
        left = PlaneSlab(point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), tol=1e-9)
        tip_face = Box((length, 0.0, 0.0), (length, side, side), tol=1e-9)
        m = LinearElasticity(
            g,
            Material(E=e, nu=nu, rho=1.0),
            supports=[(left, "all")],
            loads=[PointLoad(tip_face, (0.0, 0.0, -p))],
        )
        u = _solve(m, np.ones(g.n_elements))
        return float(-m.displacement_field(u).values[:, 2].min())

    coarse = tip((16, 2, 2))
    fine = tip((32, 4, 4))
    assert abs(fine - reference) / reference < 0.06
    assert abs(fine - reference) < abs(coarse - reference)


def test_integer_dof_spec_equivalent_to_names() -> None:
    g = StructuredGrid(shape=(2, 2), spacing=(1.0, 1.0))
    left = PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0), tol=1e-9)
    a = LinearElasticity(g, STEEL, supports=[(left, (0,))], loads=[BodyForce((0.0, -1.0))])
    b = LinearElasticity(g, STEEL, supports=[(left, "x")], loads=[BodyForce((0.0, -1.0))])
    assert a.n_dof == b.n_dof
    assert a.dof_index(0, 1) == b.dof_index(0, 1)


def test_unsupported_element_kind_rejected() -> None:
    class FakeKindGrid(StructuredGrid):
        @property
        def element_kind(self) -> str:
            return "tet4"

    g = FakeKindGrid(shape=(2, 2), spacing=(1.0, 1.0))
    with pytest.raises(FemError, match="tet4"):
        LinearElasticity(
            g, STEEL, supports=[(NearPoint((0.0, 0.0)), "all")], loads=[BodyForce((0.0, -1.0))]
        )


def test_assemble_matches_dense_reference() -> None:
    void = np.zeros(8, dtype=bool)
    void[5] = True
    solid = np.zeros(8, dtype=bool)
    solid[0] = True
    g = StructuredGrid(shape=(4, 2), spacing=(1.0, 1.5), solid=solid, void=void)
    left = PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0), tol=1e-9)
    tip = NearPoint((4.0, 1.5))
    model = LinearElasticity(
        g,
        Material(E=7.0, nu=0.28, rho=1.0),
        supports=[(left, "all")],
        loads=[PointLoad(tip, (0.0, -1.0))],
    )
    rng = np.random.default_rng(3)
    scale = rng.uniform(0.2, 1.0, g.n_elements)
    k = model.assemble(scale)
    indptr, indices, data = k.csr_arrays()
    dense = scipy.sparse.csr_array(
        (np.asarray(data), np.asarray(indices), np.asarray(indptr)), shape=k.shape
    ).toarray()
    ref = np.zeros((model.n_dof, model.n_dof))
    ke = model.element_stiffness
    for e in np.flatnonzero(g.active_elements):
        dofs = [model.dof_index(int(n), c) for n in g.element_nodes[e] for c in range(2)]
        for a, da in enumerate(dofs):
            if da < 0:
                continue
            for b, db in enumerate(dofs):
                if db < 0:
                    continue
                ref[da, db] += scale[e] * ke[a, b]
    np.testing.assert_allclose(dense, ref, rtol=1e-12, atol=1e-12)


def test_near_nullspace_matches_rigid_body_fields() -> None:
    void = np.zeros(8, dtype=bool)
    void[5] = True
    g = StructuredGrid(shape=(4, 2), spacing=(1.0, 1.5), void=void)
    left = PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0), tol=1e-9)
    model = LinearElasticity(
        g,
        Material(E=1.0, nu=0.3, rho=1.0),
        supports=[(left, "all")],
        loads=[PointLoad(NearPoint((4.0, 1.5)), (0.0, -1.0))],
    )
    b = model.near_nullspace()
    assert b.shape == (model.n_dof, 3)
    assert np.linalg.matrix_rank(b) == 3
    center = g.nodes[g.active_nodes].mean(axis=0)
    for node in np.flatnonzero(g.active_nodes):
        xy = g.nodes[node] - center
        for comp in range(2):
            i = model.dof_index(int(node), comp)
            if i < 0:
                continue
            np.testing.assert_allclose(b[i, 0], 1.0 if comp == 0 else 0.0)
            np.testing.assert_allclose(b[i, 1], 0.0 if comp == 0 else 1.0)
            np.testing.assert_allclose(b[i, 2], -xy[1] if comp == 0 else xy[0])


def test_near_nullspace_3d_shape() -> None:
    g = StructuredGrid.box(size=(4.0, 2.0, 2.0), shape=(4, 2, 2))
    left = PlaneSlab(point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), tol=1e-9)
    model = LinearElasticity(
        g,
        Material(E=1.0, nu=0.3, rho=1.0),
        supports=[(left, "all")],
        loads=[PointLoad(NearPoint((4.0, 1.0, 1.0)), (0.0, -1.0, 0.0))],
    )
    b = model.near_nullspace()
    assert b.shape == (model.n_dof, 6)
    assert np.linalg.matrix_rank(b) == 6
