# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Tier-1 tests for linear elasticity: elements, loads, patch and analytic cases."""

import numpy as np
import pytest
import scipy.sparse

from tests.conftest import LEFT_EDGE_2D, LEFT_EDGE_3D, ones, solve
from topokit.fem import (
    ALUMINUM_6061,
    STEEL,
    BodyForce,
    FemError,
    LinearElasticity,
    Material,
    PointLoad,
    SurfaceTraction,
)
from topokit.mesh import StructuredGrid
from topokit.selection import Box, NearPoint, OnBoundary, PlaneSlab


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
    m = LinearElasticity(
        g,
        STEEL,
        supports=[(LEFT_EDGE_2D, "all")],
        loads=[PointLoad(NearPoint((1.0, 0.5)), (1.0, 0.0))],
    )
    # 4 active nodes, 2 fully fixed -> 4 free dofs
    assert m.n_dof == 4


def test_traction_consistent_nodal_values() -> None:
    g = StructuredGrid(shape=(2, 2), spacing=(1.0, 1.0))
    right = Box((2.0, 0.0), (2.0, 2.0), tol=1e-9)
    m = LinearElasticity(
        g,
        STEEL,
        supports=[(LEFT_EDGE_2D, "all")],
        loads=[SurfaceTraction(right, traction=(10.0, 0.0))],
    )
    f = m.loads()[:, 0]
    assert f[m.dof_index(5, 0)] == pytest.approx(10.0)  # mid-edge node, two faces
    assert f[m.dof_index(2, 0)] == pytest.approx(5.0)
    assert f[m.dof_index(8, 0)] == pytest.approx(5.0)
    assert f.sum() == pytest.approx(20.0)  # total = t * area


def test_pressure_acts_against_outward_normal() -> None:
    g = StructuredGrid(shape=(1, 1), spacing=(1.0, 1.0))
    top = Box((0.0, 1.0), (1.0, 1.0), tol=1e-9)
    m = LinearElasticity(
        g, STEEL, supports=[(LEFT_EDGE_2D, "all")], loads=[SurfaceTraction(top, pressure=4.0)]
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
    m = LinearElasticity(
        g,
        STEEL,
        supports=[(LEFT_EDGE_2D, "all")],
        loads=[
            [PointLoad(NearPoint((2.0, 2.0)), (0.0, -1.0))],
            [PointLoad(NearPoint((2.0, 0.0)), (1.0, 0.0))],
        ],
    )
    assert m.n_cases == 2
    assert m.loads().shape == (m.n_dof, 2)


def test_uniaxial_patch_3d_exact() -> None:
    g = StructuredGrid(shape=(2, 1, 1), spacing=(1.0, 1.0, 1.0))
    e, sigma = 100.0, 10.0
    mat = Material(E=e, nu=0.3, rho=1.0)
    face_x2 = Box((2.0, 0.0, 0.0), (2.0, 1.0, 1.0), tol=1e-9)
    supports = [
        (LEFT_EDGE_3D, "x"),
        (NearPoint((0.0, 0.0, 0.0)), ("y", "z")),
        (NearPoint((0.0, 0.0, 1.0)), ("y",)),
    ]
    m = LinearElasticity(
        g, mat, supports=supports, loads=[SurfaceTraction(face_x2, traction=(sigma, 0, 0))]
    )
    scale = ones(g)
    u = solve(m, scale)
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
        right = Box((4.0, 0.0), (4.0, 2.0), tol=1e-9)
        m = LinearElasticity(
            g,
            mat,
            supports=[(LEFT_EDGE_2D, "x"), (NearPoint((0.0, 0.0)), "y")],
            loads=[SurfaceTraction(right, traction=(10.0, 0.0))],
            mode=mode,
        )
        u = solve(m, ones(g))
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
        u = solve(m, np.ones(shape[0] * shape[1]))
        return float(-m.displacement_field(u).values[:, 1].min())

    coarse = tip_deflection((24, 4))
    fine = tip_deflection((48, 8))
    assert abs(fine - reference) / reference < 0.05
    assert abs(fine - reference) < abs(coarse - reference)  # converging


def test_void_elements_have_zero_energy_and_stress() -> None:
    void = np.zeros(8, dtype=bool)
    void[7] = True
    g = StructuredGrid(shape=(4, 2), spacing=(1.0, 1.0), void=void)
    m = LinearElasticity(
        g,
        STEEL,
        supports=[(LEFT_EDGE_2D, "all")],
        loads=[PointLoad(NearPoint((4.0, 0.0)), (0.0, -1.0))],
    )
    scale = ones(g)
    u = solve(m, scale)
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
    u = solve(m, ones(g))
    np.testing.assert_allclose(m.element_stress(u, ones(g)), 10.0, rtol=1e-9)


def test_uniaxial_patch_with_solid_region_exact() -> None:
    solid = np.zeros(8, dtype=bool)
    solid[[1, 5]] = True  # one full column of solid elements
    g = StructuredGrid(shape=(4, 2), spacing=(1.0, 1.0), solid=solid)
    mat = Material(E=100.0, nu=0.3, rho=1.0)
    right = Box((4.0, 0.0), (4.0, 2.0), tol=1e-9)
    m = LinearElasticity(
        g,
        mat,
        supports=[(LEFT_EDGE_2D, "x"), (NearPoint((0.0, 0.0)), "y")],
        loads=[SurfaceTraction(right, traction=(10.0, 0.0))],
    )
    u = solve(m, ones(g))
    np.testing.assert_allclose(m.element_stress(u, ones(g)), 10.0, rtol=1e-9)


def test_plane_strain_von_mises_includes_sigma_z() -> None:
    g = StructuredGrid(shape=(4, 2), spacing=(1.0, 1.0))
    mat = Material(E=100.0, nu=0.3, rho=1.0)
    right = Box((4.0, 0.0), (4.0, 2.0), tol=1e-9)
    m = LinearElasticity(
        g,
        mat,
        supports=[(LEFT_EDGE_2D, "x"), (NearPoint((0.0, 0.0)), "y")],
        loads=[SurfaceTraction(right, traction=(10.0, 0.0))],
        mode="plane_strain",
    )
    u = solve(m, ones(g))
    # sigma_z = nu*(sx+sy) = 3, vm = sqrt(0.5*((10)^2 + 3^2 + 7^2)) = sqrt(79)
    np.testing.assert_allclose(m.element_stress(u, ones(g)), np.sqrt(79.0), rtol=1e-9)


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
    u = solve(m, scale)
    energy = float(u @ k.matvec(u))
    work = float(m.loads()[:, 0] @ u)
    assert energy == pytest.approx(work, rel=1e-4)


def test_pressure_on_void_interface_face() -> None:
    g = StructuredGrid(shape=(2, 1), spacing=(1.0, 1.0), void=[False, True])
    interface = Box((1.0, 0.0), (1.0, 1.0), tol=1e-9)
    m = LinearElasticity(
        g,
        STEEL,
        supports=[(LEFT_EDGE_2D, "all")],
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
        tip_face = Box((length, 0.0, 0.0), (length, side, side), tol=1e-9)
        m = LinearElasticity(
            g,
            Material(E=e, nu=nu, rho=1.0),
            supports=[(LEFT_EDGE_3D, "all")],
            loads=[PointLoad(tip_face, (0.0, 0.0, -p))],
        )
        u = solve(m, np.ones(g.n_elements))
        return float(-m.displacement_field(u).values[:, 2].min())

    coarse = tip((16, 2, 2))
    fine = tip((32, 4, 4))
    assert abs(fine - reference) / reference < 0.06
    assert abs(fine - reference) < abs(coarse - reference)


def test_integer_dof_spec_equivalent_to_names() -> None:
    g = StructuredGrid(shape=(2, 2), spacing=(1.0, 1.0))
    a = LinearElasticity(g, STEEL, supports=[(LEFT_EDGE_2D, (0,))], loads=[BodyForce((0.0, -1.0))])
    b = LinearElasticity(g, STEEL, supports=[(LEFT_EDGE_2D, "x")], loads=[BodyForce((0.0, -1.0))])
    assert a.n_dof == b.n_dof
    assert a.dof_index(0, 1) == b.dof_index(0, 1)


def test_assemble_matches_dense_reference() -> None:
    void = np.zeros(8, dtype=bool)
    void[5] = True
    solid = np.zeros(8, dtype=bool)
    solid[0] = True
    g = StructuredGrid(shape=(4, 2), spacing=(1.0, 1.5), solid=solid, void=void)
    tip = NearPoint((4.0, 1.5))
    model = LinearElasticity(
        g,
        Material(E=7.0, nu=0.28, rho=1.0),
        supports=[(LEFT_EDGE_2D, "all")],
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
    model = LinearElasticity(
        g,
        Material(E=1.0, nu=0.3, rho=1.0),
        supports=[(LEFT_EDGE_2D, "all")],
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
    model = LinearElasticity(
        g,
        Material(E=1.0, nu=0.3, rho=1.0),
        supports=[(LEFT_EDGE_3D, "all")],
        loads=[PointLoad(NearPoint((4.0, 1.0, 1.0)), (0.0, -1.0, 0.0))],
    )
    b = model.near_nullspace()
    assert b.shape == (model.n_dof, 6)
    assert np.linalg.matrix_rank(b) == 6


def test_a_per_element_stack_assembles_exactly_like_the_shared_matrix() -> None:
    """`ElasticityBase`'s hook may return either stiffness layout.

    One assembly kernel serves both because the base hands it the stiffness
    pair-major, so a grid's single matrix and a per-element stack carrying
    copies of it must produce the same CSR data bit for bit. This is also the
    smallest `ElasticityBase` subclass in the suite: one overridden hook.
    """

    class StackedElasticity(LinearElasticity):
        def _build_elements(
            self, conn: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            ke, b, face_w, body_w = super()._build_elements(conn)
            n_ae = conn.shape[0]
            return (
                np.broadcast_to(ke, (n_ae, *ke.shape)).copy(),
                np.broadcast_to(b, (n_ae, *b.shape)).copy(),
                face_w,
                body_w,
            )

    void = np.zeros(8, dtype=bool)
    void[5] = True
    g = StructuredGrid(shape=(4, 2), spacing=(1.0, 1.5), void=void)
    supports = [(LEFT_EDGE_2D, "all")]
    loads = [SurfaceTraction(Box((4.0, 0.0), (4.0, 3.0), tol=1e-9), traction=(1.0, -2.0))]
    shared = LinearElasticity(g, STEEL, supports, loads)
    stacked = StackedElasticity(g, STEEL, supports, loads)

    rng = np.random.default_rng(20261018)
    scale = rng.uniform(0.2, 1.0, g.n_elements)
    u = rng.normal(size=shared.n_dof)
    np.testing.assert_array_equal(shared.loads(), stacked.loads())
    np.testing.assert_array_equal(
        np.asarray(shared.assemble(scale).csr_arrays()[2]),
        np.asarray(stacked.assemble(scale).csr_arrays()[2]),
    )
    # the energy and stress contractions broadcast over both layouts; BLAS
    # reassociates between them, so they agree to rounding, not bit for bit
    np.testing.assert_allclose(
        shared.element_energies(u, scale), stacked.element_energies(u, scale), rtol=1e-12
    )
    np.testing.assert_allclose(
        shared.element_stress(u, scale), stacked.element_stress(u, scale), rtol=1e-12
    )
