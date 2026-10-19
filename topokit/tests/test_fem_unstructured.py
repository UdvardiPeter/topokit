# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Tier-1 tests for ``UnstructuredElasticity`` beyond the conformance bar.

The checks both elasticity models owe the `PhysicsModel` contract live in
`test_physics_conformance.py`, parametrized over quad4/tri3/tet4/tet10.
What is here is what only the simplex models can be asked: the quadratic
tet10 states, the consistent-load closed forms, the void-element
condensation, and the end-to-end FD and `Study` runs on non-grid meshes.

Meshes come from `tests.conftest` (hand-built, no mesh-generator
dependency): the 2-triangle unit square, the quad-split tri3 grid, and the
Kuhn triangulation of a cube lattice into 6 tet4s, optionally upgraded to
tet10 by one midpoint node per unique edge.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.conftest import (
    CORNER_TET10_NODES,
    LEFT_EDGE_2D,
    LEFT_EDGE_3D,
    UNIT_MATERIAL,
    corner_tet10,
    ones,
    solve,
    tet4_box,
    tet10_box,
    tri3_grid,
    tri3_square,
)
from topokit.fem import (
    BodyForce,
    FemError,
    LinearElasticity,
    Material,
    PointLoad,
    SurfaceTraction,
    UnstructuredElasticity,
)
from topokit.mesh import StructuredGrid, UnstructuredMesh
from topokit.optimizers import MMA, OC
from topokit.parametrization import SIMP, DensityFilter
from topokit.problem import Problem, Schedule, Study
from topokit.responses import Compliance, Solution, Volume
from topokit.selection import Box, NearPoint, PlaneSlab
from topokit.solvers import Direct
from topokit.testing import assert_gradient_matches


def _face_consistent_linear_loads(
    corner_coords: np.ndarray, t_corner: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Consistent tet10-face nodal loads for a traction linear across the face.

    ``t_corner`` gives the (scalar) traction magnitude at each of the 3
    corners; the field is ``t(x) = sum_k t_corner[k] * L_k(x)`` (the area
    coordinates), matching a traction affine in the physical coordinates on
    a straight triangular face. Derived from the same barycentric-moment
    formula ``test_elements.py`` uses for the energy patch tests
    (``int L1^a L2^b L3^c dA = 2A a!b!c!/(a+b+c+2)!``); the constant-traction
    special case (``t_corner`` all equal) reduces exactly to
    ``fem._elements.face_weights("tet10")`` (0 at corners, area/3 at
    midsides), cross-checking the derivation.

    Returns ``(corner_loads, midside_loads, area)``: 3 corner values (order
    matching ``corner_coords``) and 3 midside values in
    ``BoundaryFaces.nodes``' midside order (edges (0,1),(1,2),(2,0)).
    """
    t1, t2, t3 = t_corner
    total = t1 + t2 + t3
    e1, e2 = corner_coords[1] - corner_coords[0], corner_coords[2] - corner_coords[0]
    area = 0.5 * float(np.linalg.norm(np.cross(e1, e2)))
    corner_loads = (area / 60.0) * (3.0 * t_corner - total)
    midside_loads = np.array(
        [
            (2.0 * area / 15.0) * (t1 + t2) + (area / 15.0) * t3,
            (2.0 * area / 15.0) * (t2 + t3) + (area / 15.0) * t1,
            (2.0 * area / 15.0) * (t3 + t1) + (area / 15.0) * t2,
        ]
    )
    return corner_loads, midside_loads, area


def test_tet10_pure_bending_quadratic_exact() -> None:
    """A pure-bending state is a complete quadratic displacement field.

    Closed-form 3D elasticity solution for pure bending about the z-axis
    (curvature ``kappa``, stress ``sigma_xx = E*kappa*y`` only):
    ``u = (kappa*x*y, -kappa/2*(x^2 + nu*(y^2 - z^2)), -kappa*nu*y*z)``.
    Every component is a complete quadratic polynomial, exactly in the span
    of tet10 shape functions, so an assembly of straight-sided tet10
    elements should reproduce it to solver precision (not just mesh-limited
    accuracy). Verified by direct construction: `exx=du_x/dx=kappa*y`,
    `eyy=ezz=-nu*kappa*y` (Poisson contraction), all shears zero.

    Essential BCs use only what the API supports (homogeneous Dirichlet):
    a roller (``ux=0``) on the whole ``x=0`` face matches ``u_x(0,y,z)=0``
    identically and also kills the y/z-axis rigid rotations (it constrains
    an entire non-degenerate face, not a point); two extra point pins (all
    3 components at the origin, where the exact field is exactly zero; the
    z-component at ``(0,1,0)``, where ``u_z=-kappa*nu*y*z`` is zero because
    ``z=0``) remove the remaining 3 translation/x-rotation gauge freedoms.
    The traction on ``x=1`` (``sigma_xx=E*kappa*y``, linear across the
    face) is applied as explicit per-node ``PointLoad``s using the
    consistent-load formula above -- ``SurfaceTraction`` only supports a
    uniform traction per selector, not one varying across a single face.
    """
    mesh = tet10_box(1, 1, 1)
    e, nu, kappa = 100.0, 0.3, 1.0
    mat = Material(E=e, nu=nu, rho=1.0)

    def u_exact(coord: np.ndarray) -> np.ndarray:
        x, y, z = coord
        return np.array(
            [
                kappa * x * y,
                -kappa / 2.0 * (x**2 + nu * (y**2 - z**2)),
                -kappa * nu * y * z,
            ]
        )

    bf = mesh.boundary_faces()
    face_ids = np.nonzero(np.isclose(bf.centroid[:, 0], 1.0))[0]
    assert face_ids.size == 2  # the x=1 face, split in 2 by the Kuhn diagonal

    accum: dict[int, float] = {}
    for fid in face_ids:
        row = bf.nodes[fid]
        corner_ids, mid_ids = row[:3], row[3:]
        t_corner = e * kappa * mesh.nodes[corner_ids][:, 1]  # sigma_xx = E*kappa*y at corners
        corner_loads, mid_loads, _area = _face_consistent_linear_loads(
            mesh.nodes[corner_ids], t_corner
        )
        for nid, val in zip([*corner_ids, *mid_ids], [*corner_loads, *mid_loads], strict=True):
            accum[int(nid)] = accum.get(int(nid), 0.0) + float(val)

    loads = [
        PointLoad(NearPoint(tuple(mesh.nodes[nid])), (force, 0.0, 0.0))
        for nid, force in accum.items()
        if abs(force) > 1e-12
    ]
    supports = [
        (LEFT_EDGE_3D, "x"),
        (NearPoint((0.0, 0.0, 0.0)), "all"),
        (NearPoint((0.0, 1.0, 0.0)), "z"),
    ]
    m = UnstructuredElasticity(mesh, mat, supports=supports, loads=loads)
    u = solve(m, ones(mesh))
    disp = m.displacement_field(u)
    for i, coord in enumerate(mesh.nodes):
        np.testing.assert_allclose(disp.values[i], u_exact(coord), atol=1e-9)


# ---------------------------------------------------------------------------
# Rigid-body / near-nullspace
# ---------------------------------------------------------------------------


def _rigid_mode_energy_ratio(m: UnstructuredElasticity, mesh: UnstructuredMesh, seed: int) -> float:
    """Max rigid-body-mode strain energy relative to a generic vector's.

    ``near_nullspace()`` is a *near*-nullspace hint: with a support in
    place, the modes are only approximately in the assembled matrix's
    nullspace (exactly zero far from the support, perturbed only near it,
    since the mesh is locally connected). Measuring modes vs. a generic
    random vector's strain energy makes that "approximately" precise
    without demanding an exact (and generally false) zero.
    """
    k = m.assemble(ones(mesh))
    b = m.near_nullspace()
    rng = np.random.default_rng(seed)
    generic = rng.normal(size=m.n_dof)
    generic_energy = float(generic @ k.matvec(generic))
    ratios = [float(b[:, j] @ k.matvec(b[:, j])) / generic_energy for j in range(b.shape[1])]
    return max(ratios)


def test_near_nullspace_lies_in_numerical_nullspace_2d() -> None:
    # a single fully-pinned node is barely enough to solve; the rest of the
    # domain is otherwise unsupported, so the rigid modes should carry far
    # less strain energy than a generic vector (empirically ~0.06-0.11 for
    # this mesh family across sizes; 0.3 leaves ample margin without being
    # toothless). A too-small mesh (e.g. the 2-triangle square) makes the
    # single pin a large fraction of the domain, so use a bigger grid.
    mesh = tri3_grid(6, 6)
    m = UnstructuredElasticity(
        mesh,
        UNIT_MATERIAL,
        supports=[(NearPoint((0.0, 0.0)), "all")],
        loads=[SurfaceTraction(Box((6.0, 0.0), (6.0, 6.0), tol=1e-9), traction=(1.0, 0.0))],
    )
    assert _rigid_mode_energy_ratio(m, mesh, seed=0) < 0.3


def test_near_nullspace_lies_in_numerical_nullspace_3d() -> None:
    mesh = tet4_box(3, 3, 3)
    pin = NearPoint((0.0, 0.0, 0.0))
    m = UnstructuredElasticity(
        mesh,
        UNIT_MATERIAL,
        supports=[(pin, "all")],
        loads=[
            SurfaceTraction(
                Box((3.0, 0.0, 0.0), (3.0, 3.0, 3.0), tol=1e-9), traction=(1.0, 0.0, 0.0)
            )
        ],
    )
    # observed ~0.01 for this mesh size, shrinking with mesh size (the pin's
    # effect is local); 0.05 leaves a comfortable margin
    assert _rigid_mode_energy_ratio(m, mesh, seed=0) < 0.05


# ---------------------------------------------------------------------------
# Constant-traction consistency (tet10)
# ---------------------------------------------------------------------------


def test_tet10_constant_traction_zero_corner_and_area_third_midside() -> None:
    # single right-angle tet10, corners at the origin and unit axis points;
    # face (1,2,3) is the hypotenuse (nodes 1,2,3 + their 3 midsides)
    mesh = corner_tet10()
    face = PlaneSlab(point=(1.0, 0.0, 0.0), normal=(1.0, 1.0, 1.0), tol=1e-6)
    traction = (0.0, 0.0, 3.0)
    m = UnstructuredElasticity(
        mesh,
        UNIT_MATERIAL,
        supports=[(NearPoint((0.0, 0.0, 0.0)), "all")],
        loads=[SurfaceTraction(face, traction=traction)],
    )
    f = m.loads()[:, 0]
    bf = mesh.boundary_faces()
    (fid,) = face.faces(mesh)
    area = float(bf.area[fid])
    for corner in (1, 2, 3):
        i = m.dof_index(corner, 2)
        assert i >= 0  # not pinned (only node 0 is)
        assert f[i] == pytest.approx(0.0, abs=1e-12)
    for midside in (5, 8, 9):  # mid(1,2), mid(1,3), mid(2,3): the (1,2,3) face's edges
        i = m.dof_index(midside, 2)
        assert f[i] == pytest.approx(traction[2] * area / 3.0)


# ---------------------------------------------------------------------------
# Body-force consistency (tet10)
# ---------------------------------------------------------------------------


def test_tet10_body_force_consistent_weights_on_free_dofs() -> None:
    """A uniform body force on a single tet10 matches the derived -V/20 / V/5 split.

    Consistent (shape-function-integral) nodal weights for a uniform body
    force are -V/20 per vertex (negative -- the well-known quadratic-tet
    result) and V/5 per midside; see ``fem._elements.body_weights``'s module
    comment for the barycentric-moment derivation. Only node 0 is pinned
    (satisfying the "at least one support" requirement without touching any
    other node), so every other node's load-vector entry is a free DOF and
    is checked directly against the closed form -- no assembly/solve
    involved, this is the load vector before any system reduction beyond
    that one pin.
    """
    mesh = corner_tet10()
    volume = 1.0 / 6.0
    b = np.array([2.0, -3.0, 5.0])
    m = UnstructuredElasticity(
        mesh,
        UNIT_MATERIAL,
        supports=[(NearPoint((0.0, 0.0, 0.0)), "all")],
        loads=[BodyForce((float(b[0]), float(b[1]), float(b[2])))],
    )
    f = m.loads()[:, 0]
    for vertex in (1, 2, 3):  # node 0 is pinned; the other 3 vertices are free
        for comp in range(3):
            i = m.dof_index(vertex, comp)
            assert i >= 0
            assert f[i] == pytest.approx(-volume / 20.0 * b[comp])
    for midside in range(4, 10):
        for comp in range(3):
            i = m.dof_index(midside, comp)
            assert i >= 0
            assert f[i] == pytest.approx(volume / 5.0 * b[comp])


# ---------------------------------------------------------------------------
# Void elements: condensation out of the system
# ---------------------------------------------------------------------------


def test_void_elements_leave_the_system_tri3() -> None:
    """Voiding the right half must give the same answer as never meshing it.

    Every per-element array the model keeps (``_ke``, ``_centroid_b``,
    ``_edofs_free``) is indexed over ACTIVE elements
    while the public ones (``element_energies``, ``element_stress``,
    ``scale``) are indexed over all elements, so a mesh with void elements
    is the only thing that exercises the two numberings against each other.
    """
    full = tri3_grid(2, 1)
    void = np.zeros(full.n_elements, dtype=bool)
    void[2:] = True  # the 2 triangles of the right cell, x in [1, 2]
    mesh = UnstructuredMesh(full.nodes, full.element_nodes, "tri3", void=void)
    assert mesh.active_nodes.sum() == 4  # the x=2 nodes drop out entirely

    e, sigma = 100.0, 10.0
    mat = Material(E=e, nu=0.3, rho=1.0)
    right = Box((1.0, 0.0), (1.0, 1.0), tol=1e-9)
    m = UnstructuredElasticity(
        mesh,
        mat,
        supports=[(LEFT_EDGE_2D, "x"), (NearPoint((0.0, 0.0)), "y")],
        loads=[SurfaceTraction(right, traction=(sigma, 0.0))],
    )
    assert m.n_dof == 2 * 4 - 3  # 4 active nodes, 2 x-pins + 1 y-pin
    scale = ones(mesh)
    u = solve(m, scale)

    # the active half is the uniaxial patch test again, exactly
    np.testing.assert_allclose(m.element_stress(u, scale)[:2], sigma, rtol=1e-9)
    tip = m.displacement_field(u).values[mesh.nodes[:, 0] == 1.0, 0]
    np.testing.assert_allclose(tip, sigma * 1.0 / e, rtol=1e-9)

    # void rows are zero in both per-element outputs, and the energies still
    # sum to the work done
    np.testing.assert_array_equal(m.element_energies(u, scale)[2:], 0.0)
    np.testing.assert_array_equal(m.element_stress(u, scale)[2:], 0.0)
    assert m.element_energies(u, scale).sum() == pytest.approx(
        float(m.loads()[:, 0] @ u), rel=1e-10
    )
    # void nodes carry no displacement
    np.testing.assert_array_equal(m.displacement_field(u).values[~mesh.active_nodes], 0.0)


def test_bad_tet10_names_its_mesh_element_id_not_its_active_row() -> None:
    """``stiffness_stack`` runs over active elements only, so it must be told ids.

    Element 0 here is void, so the bad element is active row 0 but mesh
    element 1; reporting the row position would point the user at the wrong
    element (and at a perfectly good one).
    """
    good = CORNER_TET10_NODES
    # a second, detached copy whose mid(0,1) node is dragged far off the edge:
    # the corner tetrahedron stays positive (so the mesh constructor accepts
    # it) but the quadratic map folds over at the Gauss points
    nodes = np.vstack([good, good + np.array([2.0, 0.0, 0.0])])
    nodes[10 + 4] += np.array([0.0, 0.0, 5.0])
    elements = np.array([list(range(10)), list(range(10, 20))], dtype=np.int64)
    mesh = UnstructuredMesh(nodes, elements, "tet10", void=np.array([True, False]))
    with pytest.raises(FemError, match=r"element ids \[1\]"):
        UnstructuredElasticity(
            mesh,
            UNIT_MATERIAL,
            supports=[(NearPoint((2.0, 0.0, 0.0)), "all")],
            loads=[BodyForce((0.0, 0.0, -1.0))],
        )


# ---------------------------------------------------------------------------
# Construction errors: same message family as the grid model
# ---------------------------------------------------------------------------


def test_empty_supports_rejected() -> None:
    mesh = tri3_square()
    with pytest.raises(FemError, match="at least one support"):
        UnstructuredElasticity(
            mesh,
            UNIT_MATERIAL,
            supports=[],
            loads=[SurfaceTraction(Box((1.0, 0.0), (1.0, 1.0), tol=1e-9), traction=(1.0, 0.0))],
        )


def test_empty_loads_rejected() -> None:
    mesh = tri3_square()
    with pytest.raises(FemError, match="at least one load"):
        UnstructuredElasticity(
            mesh,
            UNIT_MATERIAL,
            supports=[(NearPoint((0.0, 0.0)), "all")],
            loads=[],
        )


# ---------------------------------------------------------------------------
# End-to-end integration: FD, convergence, tet10 vs tet4, and Study smoke runs
# ---------------------------------------------------------------------------


def test_compliance_gradient_fd_through_full_stack_tri3() -> None:
    """Mirrors ``test_responses.test_compliance_gradient_fd_through_full_stack``.

    ``DensityFilter`` on a non-grid mesh dispatches to its sparse-matrix path
    (``_BoundMatrixFilter``); this is the first FD check of that path chained
    all the way through ``UnstructuredElasticity`` and ``Direct``.
    """
    mesh = tri3_grid(4, 3)
    mat = Material(E=100.0, nu=0.3, rho=1.0)
    tip = Box((4.0, 0.0), (4.0, 3.0), tol=1e-9)
    model = UnstructuredElasticity(
        mesh,
        mat,
        supports=[(LEFT_EDGE_2D, "all")],
        loads=[SurfaceTraction(tip, traction=(1.0, 0.0))],
    )
    chain = (DensityFilter(radius=1.5) | SIMP(p=3.0)).bind(mesh)
    rng = np.random.default_rng(20260823)
    x = rng.uniform(0.3, 0.9, size=chain.n_vars)  # random interior densities
    comp = Compliance()

    def _solution(xx: np.ndarray) -> Solution:
        scale = chain.apply(xx)
        rho = chain.physical_density(xx)
        solver = Direct()
        solver.prepare(model.assemble(scale))
        u = np.atleast_2d(np.asarray(solver.solve(model.loads()))).reshape(model.n_dof, -1)
        return Solution(model=model, mesh=mesh, displacements=u, interpolated=scale, density=rho)

    def f(xx: np.ndarray) -> float:
        return comp.value(_solution(xx))

    def grad(xx: np.ndarray) -> np.ndarray:
        sol = _solution(xx)
        return np.asarray(chain.pullback(xx, comp.grad_field(sol)))

    assert_gradient_matches(f, grad, x, rtol=1e-5)


def test_compliance_gradient_fd_through_full_stack_tet10() -> None:
    """The same FD check on a quadratic tet mesh (a 1x1x1 Kuhn box, 6 tet10s).

    The adjoint is only right if the ``_ke`` stack that ``element_energies``
    contracts is exactly the one ``assemble`` scatters, and tet10 is the one
    element whose stack comes out of a quadrature loop rather than a closed
    form -- so it needs its own check, not just tri3's.
    """
    mesh = tet10_box(1, 1, 1)
    mat = Material(E=100.0, nu=0.3, rho=1.0)
    face_x1 = Box((1.0, 0.0, 0.0), (1.0, 1.0, 1.0), tol=1e-9)
    model = UnstructuredElasticity(
        mesh,
        mat,
        supports=[(LEFT_EDGE_3D, "all")],
        loads=[SurfaceTraction(face_x1, traction=(1.0, 0.0, -1.0))],
    )
    chain = (DensityFilter(radius=1.5) | SIMP(p=3.0)).bind(mesh)
    rng = np.random.default_rng(20260824)
    x = rng.uniform(0.3, 0.9, size=chain.n_vars)
    comp = Compliance()

    def _solution(xx: np.ndarray) -> Solution:
        scale = chain.apply(xx)
        solver = Direct()
        solver.prepare(model.assemble(scale))
        u = np.atleast_2d(np.asarray(solver.solve(model.loads()))).reshape(model.n_dof, -1)
        return Solution(
            model=model,
            mesh=mesh,
            displacements=u,
            interpolated=scale,
            density=chain.physical_density(xx),
        )

    assert_gradient_matches(
        lambda xx: comp.value(_solution(xx)),
        lambda xx: np.asarray(chain.pullback(xx, comp.grad_field(_solution(xx)))),
        x,
        rtol=1e-5,
    )


def test_tri3_cantilever_converges_toward_timoshenko() -> None:
    """Same reference as ``test_fem.test_cantilever_matches_timoshenko``.

    tri3 (constant-strain) elements lock in bending, so convergence is slower
    than the hex grid's; the assertion is monotonic convergence (finer mesh
    closer to the reference), not a fixed tolerance -- the same shape as the
    grid test.
    """
    e, nu, length, height, p = 1000.0, 0.3, 6.0, 1.0, 1.0
    inertia = height**3 / 12.0
    shear_g = e / (2.0 * (1.0 + nu))
    kappa = 5.0 / 6.0
    reference = p * length**3 / (3.0 * e * inertia) + p * length / (kappa * shear_g * height)

    def tip_deflection(nx: int, ny: int) -> float:
        mesh = tri3_grid(nx, ny, (length, height))
        mat = Material(E=e, nu=nu, rho=1.0)
        tip = Box((length, 0.0), (length, height), tol=1e-9)
        m = UnstructuredElasticity(
            mesh, mat, supports=[(LEFT_EDGE_2D, "all")], loads=[PointLoad(tip, (0.0, -p))]
        )
        u = solve(m, ones(mesh))
        disp = m.displacement_field(u)
        tip_nodes = np.isclose(mesh.nodes[:, 0], length)
        return float(-disp.values[tip_nodes, 1].min())

    coarse = tip_deflection(24, 4)
    fine = tip_deflection(48, 8)
    assert abs(fine - reference) < abs(coarse - reference)  # converging
    assert abs(fine - reference) / reference < 0.1


def test_tet10_closer_than_tet4_at_equal_dof_budget() -> None:
    """tet10 (quadratic) beats tet4 (linear) at matched DOF on one cantilever.

    Both meshes cover the same 6x2x2 box (the builders' ``size=``): tet4 is
    finely subdivided (13x2x2 Kuhn cubes, 312 elements, 378 DOF) and tet10 is
    coarsely subdivided (2x2x2 cubes, 48 elements, 375 DOF) -- an equal-DOF
    pair found by search over small Kuhn grids. The reference is a fine-grid
    ``LinearElasticity`` (hex) solve on the same domain/BCs, not the
    Timoshenko closed form: the tet10 cubes are stretched 3:1 along the beam
    axis, so a beam formula would conflate discretization error with that
    element-shape choice. Measured errors are ~5% (tet10) vs ~39% (tet4)
    against the hex reference -- a wide, stable margin, not a close call.
    """
    e, nu, p = 1000.0, 0.3, 1.0
    length, side = 6.0, 2.0
    size = (length, side, side)
    mat = Material(E=e, nu=nu, rho=1.0)
    tip_sel = Box((length, 0.0, 0.0), (length, side, side), tol=1e-9)
    tip_load = [PointLoad(tip_sel, (0.0, 0.0, -p))]

    def tip_deflection(model: Any) -> float:
        u = solve(model, np.ones(model.mesh.n_elements))
        disp = model.displacement_field(u)
        tip_nodes = np.isclose(model.mesh.nodes[:, 0], length)
        return float(-disp.values[tip_nodes, 2].min())

    g = StructuredGrid.box(size=size, shape=(24, 8, 8))
    hex_model = LinearElasticity(g, mat, supports=[(LEFT_EDGE_3D, "all")], loads=tip_load)
    reference = tip_deflection(hex_model)

    tet4_mesh = tet4_box(13, 2, 2, size)
    tet10_mesh = tet10_box(2, 2, 2, size)
    assert tet4_mesh.n_nodes * 3 == pytest.approx(tet10_mesh.n_nodes * 3, rel=0.02)  # equal DOF

    tet4_model = UnstructuredElasticity(
        tet4_mesh, mat, supports=[(LEFT_EDGE_3D, "all")], loads=tip_load
    )
    tet10_model = UnstructuredElasticity(
        tet10_mesh, mat, supports=[(LEFT_EDGE_3D, "all")], loads=tip_load
    )
    tet4_tip = tip_deflection(tet4_model)
    tet10_tip = tip_deflection(tet10_model)

    assert abs(tet10_tip - reference) < abs(tet4_tip - reference)


def test_study_smoke_tri3_oc() -> None:
    """Full ``Study`` smoke run on a tri3 cantilever."""
    nx, ny = 8, 4
    mesh = tri3_grid(nx, ny)
    tip = Box((float(nx), 0.0), (float(nx), float(ny)), tol=1e-9)
    model = UnstructuredElasticity(
        mesh, UNIT_MATERIAL, supports=[(LEFT_EDGE_2D, "all")], loads=[PointLoad(tip, (0.0, -1.0))]
    )
    problem = Problem(
        model,
        DensityFilter(radius=1.5) | SIMP(p=3.0),
        objective=Compliance(),
        constraints=[Volume() <= 0.5],
        optimizer=OC(),
    )
    result = Study(problem, schedule=Schedule.single(max_iter=25, tol=1e-3)).run()

    assert result.objective < result.history["objective"][0]
    assert abs(result.history["volume"][-1] - 0.5) < 0.01
    assert result.converged or result.iterations == 25
    assert (result.design.values >= 0.0).all() and (result.design.values <= 1.0).all()


def test_study_smoke_tet4_mma() -> None:
    """Full ``Study`` smoke run on the 2-tet-refined tet4 box, MMA, 15 iterations."""
    mesh = tet4_box(2, 2, 2)
    tip = Box((2.0, 0.0, 0.0), (2.0, 2.0, 2.0), tol=1e-9)
    model = UnstructuredElasticity(
        mesh,
        UNIT_MATERIAL,
        supports=[(LEFT_EDGE_3D, "all")],
        loads=[PointLoad(tip, (0.0, 0.0, -1.0))],
    )
    problem = Problem(
        model,
        DensityFilter(radius=1.5) | SIMP(p=3.0),
        objective=Compliance(),
        constraints=[Volume() <= 0.5],
        optimizer=MMA(),
    )
    result = Study(problem, schedule=Schedule.single(max_iter=15, tol=1e-3)).run()

    assert result.objective < result.history["objective"][0]
    assert abs(result.history["volume"][-1] - 0.5) < 0.01
    assert result.converged or result.iterations == 15
    assert (result.design.values >= 0.0).all() and (result.design.values <= 1.0).all()
