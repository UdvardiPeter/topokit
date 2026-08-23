# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Tier-1 tests for ``UnstructuredElasticity``, mirroring the grid model's bar.

Mesh fixtures are built by hand (no mesh generator dependency):

- ``_tri3_square``: the 2-triangle unit square (same split as
  ``test_mesh_conformance``'s ``tri3_square`` fixture).
- ``_tet4_box``: a Kuhn (Freudenthal) triangulation of an ``nx*ny*nz`` grid
  of unit cubes into 6 tets each, sharing corner nodes -- a valid conforming
  tet4 mesh with no extra (non-corner) nodes.
- ``_tet10_box``: ``_tet4_box`` upgraded to tet10 by inserting an edge
  midpoint node per unique edge (shared between adjacent elements).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from topokit.fem import (
    BodyForce,
    FemError,
    LinearElasticity,
    Material,
    PhysicsModel,
    PointLoad,
    SurfaceTraction,
    UnstructuredElasticity,
)
from topokit.fields import FieldSpec
from topokit.mesh import StructuredGrid, UnstructuredMesh
from topokit.optimizers import MMA, OC
from topokit.parametrization import SIMP, DensityFilter
from topokit.problem import Problem, Schedule, Study
from topokit.responses import Compliance, Solution, Volume
from topokit.selection import Box, NearPoint, PlaneSlab
from topokit.solvers import Direct
from topokit.testing import assert_gradient_matches

_TET10_MIDSIDE_PAIRS = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))

# Kuhn decomposition of a unit cube into 6 tets, corners only (see
# module docstring): for each of the 3! axis orderings, walk the cube
# diagonal one axis at a time. Local corner order is
# (0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,0,1),(1,0,1),(0,1,1),(1,1,1); the
# element rows below are reordered from the raw permutation walk so every
# tet has positive signed volume (`UnstructuredMesh` requires it).
_KUHN_CUBE_LOCAL = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (1, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
)
_KUHN_CUBE_TETS = (
    (0, 1, 3, 7),
    (0, 1, 7, 5),
    (0, 2, 7, 3),
    (0, 2, 6, 7),
    (0, 4, 5, 7),
    (0, 4, 7, 6),
)


def _tri3_square() -> UnstructuredMesh:
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    elements = np.array([[0, 1, 2], [0, 2, 3]])
    return UnstructuredMesh(nodes, elements, "tri3")


def _tet4_box_arrays(nx: int, ny: int, nz: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys, zs = np.arange(nx + 1), np.arange(ny + 1), np.arange(nz + 1)
    grid = np.meshgrid(xs, ys, zs, indexing="ij")
    nodes = np.stack([g.ravel(order="F") for g in grid], axis=1).astype(float)

    def nid(i: int, j: int, k: int) -> int:
        return i + (nx + 1) * j + (nx + 1) * (ny + 1) * k

    elements = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                corners = [nid(i + di, j + dj, k + dk) for di, dj, dk in _KUHN_CUBE_LOCAL]
                for tet in _KUHN_CUBE_TETS:
                    elements.append([corners[t] for t in tet])
    return nodes, np.array(elements, dtype=np.int64)


def _tet4_box(nx: int, ny: int, nz: int) -> UnstructuredMesh:
    nodes, elements = _tet4_box_arrays(nx, ny, nz)
    return UnstructuredMesh(nodes, elements, "tet4")


def _tet4_to_tet10(nodes: np.ndarray, elements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Insert one midside node per unique edge, shared between elements."""
    node_list = [n for n in nodes]
    edge_mid: dict[frozenset[int], int] = {}
    rows = []
    for elem in elements:
        row = list(elem[:4])
        for a, b in _TET10_MIDSIDE_PAIRS:
            ga, gb = int(elem[a]), int(elem[b])
            key = frozenset((ga, gb))
            if key not in edge_mid:
                edge_mid[key] = len(node_list)
                node_list.append((nodes[ga] + nodes[gb]) / 2.0)
            row.append(edge_mid[key])
        rows.append(row)
    return np.array(node_list), np.array(rows, dtype=np.int64)


def _tri3_grid(nx: int, ny: int) -> UnstructuredMesh:
    """A ``nx*ny`` unit-square grid, each cell split into 2 triangles."""
    xs, ys = np.arange(nx + 1), np.arange(ny + 1)
    grid = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.stack([g.ravel(order="F") for g in grid], axis=1).astype(float)

    def nid(i: int, j: int) -> int:
        return i + (nx + 1) * j

    elements = []
    for i in range(nx):
        for j in range(ny):
            n0, n1, n2, n3 = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
            elements.append([n0, n1, n2])
            elements.append([n0, n2, n3])
    return UnstructuredMesh(nodes, np.array(elements, dtype=np.int64), "tri3")


def _tet10_box(nx: int, ny: int, nz: int) -> UnstructuredMesh:
    nodes4, elements4 = _tet4_box_arrays(nx, ny, nz)
    nodes10, elements10 = _tet4_to_tet10(nodes4, elements4)
    return UnstructuredMesh(nodes10, elements10, "tet10")


def _solve(model: UnstructuredElasticity, scale: np.ndarray) -> np.ndarray:
    solver = Direct()
    solver.prepare(model.assemble(scale))
    return np.asarray(solver.solve(model.loads()[:, 0]))


def _ones(mesh: UnstructuredMesh) -> np.ndarray:
    return np.ones(mesh.n_elements)


# ---------------------------------------------------------------------------
# Patch tests, exact to 1e-9
# ---------------------------------------------------------------------------


def test_uniaxial_patch_2d_exact_tri3() -> None:
    mesh = _tri3_square()
    e, sigma = 100.0, 10.0
    mat = Material(E=e, nu=0.3, rho=1.0)
    left = PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0), tol=1e-9)
    right = Box((1.0, 0.0), (1.0, 1.0), tol=1e-9)
    m = UnstructuredElasticity(
        mesh,
        mat,
        supports=[(left, "x"), (NearPoint((0.0, 0.0)), "y")],
        loads=[SurfaceTraction(right, traction=(sigma, 0.0))],
    )
    scale = _ones(mesh)
    u = _solve(m, scale)
    disp = m.displacement_field(u)
    tip = disp.values[mesh.nodes[:, 0] == 1.0, 0]
    np.testing.assert_allclose(tip, sigma * 1.0 / e, rtol=1e-9)
    np.testing.assert_allclose(m.element_stress(u, scale), sigma, rtol=1e-9)


def _uniaxial_3d_setup(
    mesh: UnstructuredMesh, mat: Material, sigma: float, mode: str = "plane_stress"
) -> UnstructuredElasticity:
    face_x0 = PlaneSlab(point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), tol=1e-9)
    face_x1 = Box((1.0, 0.0, 0.0), (1.0, 1.0, 1.0), tol=1e-9)
    supports = [
        (face_x0, "x"),
        (NearPoint((0.0, 0.0, 0.0)), ("y", "z")),
        (NearPoint((0.0, 0.0, 1.0)), ("y",)),
    ]
    return UnstructuredElasticity(
        mesh,
        mat,
        supports=supports,
        loads=[SurfaceTraction(face_x1, traction=(sigma, 0.0, 0.0))],
        mode=mode,
    )


def test_uniaxial_patch_3d_exact_tet4() -> None:
    mesh = _tet4_box(1, 1, 1)
    e, sigma = 100.0, 10.0
    mat = Material(E=e, nu=0.3, rho=1.0)
    m = _uniaxial_3d_setup(mesh, mat, sigma)
    scale = _ones(mesh)
    u = _solve(m, scale)
    np.testing.assert_allclose(m.element_stress(u, scale), sigma, rtol=1e-9)
    disp = m.displacement_field(u)
    tip = disp.values[mesh.nodes[:, 0] == 1.0, 0]
    np.testing.assert_allclose(tip, sigma * 1.0 / e, rtol=1e-9)


def test_uniaxial_patch_3d_exact_tet10() -> None:
    mesh = _tet10_box(1, 1, 1)
    e, sigma = 100.0, 10.0
    mat = Material(E=e, nu=0.3, rho=1.0)
    m = _uniaxial_3d_setup(mesh, mat, sigma)
    scale = _ones(mesh)
    u = _solve(m, scale)
    np.testing.assert_allclose(m.element_stress(u, scale), sigma, rtol=1e-9)
    disp = m.displacement_field(u)
    tip = disp.values[np.isclose(mesh.nodes[:, 0], 1.0), 0]
    np.testing.assert_allclose(tip, sigma * 1.0 / e, rtol=1e-9)


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
    mesh = _tet10_box(1, 1, 1)
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
    face_x0 = PlaneSlab(point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), tol=1e-9)
    supports = [
        (face_x0, "x"),
        (NearPoint((0.0, 0.0, 0.0)), "all"),
        (NearPoint((0.0, 1.0, 0.0)), "z"),
    ]
    m = UnstructuredElasticity(mesh, mat, supports=supports, loads=loads)
    u = _solve(m, _ones(mesh))
    disp = m.displacement_field(u)
    for i, coord in enumerate(mesh.nodes):
        np.testing.assert_allclose(disp.values[i], u_exact(coord), atol=1e-9)


def test_energy_identity_tet4_mixed_bc() -> None:
    # 2x1x1 box: face_x0 roller + 2 point pins (mixed BC), traction on the
    # true far face (x=2, not the x=1 mid-box interface)
    mesh = _tet4_box(2, 1, 1)
    mat = Material(E=100.0, nu=0.3, rho=1.0)
    face_x0 = PlaneSlab(point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), tol=1e-9)
    face_x2 = Box((2.0, 0.0, 0.0), (2.0, 1.0, 1.0), tol=1e-9)
    supports = [
        (face_x0, "x"),
        (NearPoint((0.0, 0.0, 0.0)), ("y", "z")),
        (NearPoint((0.0, 0.0, 1.0)), ("y",)),
    ]
    m = UnstructuredElasticity(
        mesh, mat, supports=supports, loads=[SurfaceTraction(face_x2, traction=(5.0, 0.0, 0.0))]
    )
    scale = _ones(mesh)
    u = _solve(m, scale)
    k = m.assemble(scale)
    energy = float(u @ k.matvec(u))
    work = float(m.loads()[:, 0] @ u)
    assert energy == pytest.approx(work, rel=1e-10)
    assert m.element_energies(u, scale).sum() == pytest.approx(energy, rel=1e-10)


# ---------------------------------------------------------------------------
# Rigid-body / near-nullspace
# ---------------------------------------------------------------------------


def test_near_nullspace_shape_2d() -> None:
    mesh = _tri3_square()
    m = UnstructuredElasticity(
        mesh,
        Material(E=1.0, nu=0.3, rho=1.0),
        supports=[(NearPoint((0.0, 0.0)), "all")],
        loads=[SurfaceTraction(Box((1.0, 0.0), (1.0, 1.0), tol=1e-9), traction=(1.0, 0.0))],
    )
    assert m.near_nullspace().shape == (m.n_dof, 3)


def test_near_nullspace_shape_3d() -> None:
    mesh = _tet4_box(1, 1, 1)
    m = _uniaxial_3d_setup(mesh, Material(E=1.0, nu=0.3, rho=1.0), sigma=1.0)
    assert m.near_nullspace().shape == (m.n_dof, 6)


def _rigid_mode_energy_ratio(m: UnstructuredElasticity, mesh: UnstructuredMesh, seed: int) -> float:
    """Max rigid-body-mode strain energy relative to a generic vector's.

    ``near_nullspace()`` is a *near*-nullspace hint: with a support in
    place, the modes are only approximately in the assembled matrix's
    nullspace (exactly zero far from the support, perturbed only near it,
    since the mesh is locally connected). Measuring modes vs. a generic
    random vector's strain energy makes that "approximately" precise
    without demanding an exact (and generally false) zero.
    """
    k = m.assemble(_ones(mesh))
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
    mesh = _tri3_grid(6, 6)
    m = UnstructuredElasticity(
        mesh,
        Material(E=1.0, nu=0.3, rho=1.0),
        supports=[(NearPoint((0.0, 0.0)), "all")],
        loads=[SurfaceTraction(Box((6.0, 0.0), (6.0, 6.0), tol=1e-9), traction=(1.0, 0.0))],
    )
    assert _rigid_mode_energy_ratio(m, mesh, seed=0) < 0.3


def test_near_nullspace_lies_in_numerical_nullspace_3d() -> None:
    mesh = _tet4_box(3, 3, 3)
    pin = NearPoint((0.0, 0.0, 0.0))
    m = UnstructuredElasticity(
        mesh,
        Material(E=1.0, nu=0.3, rho=1.0),
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
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.0, 0.0],  # mid(0,1)
            [0.5, 0.5, 0.0],  # mid(1,2)
            [0.0, 0.5, 0.0],  # mid(2,0)
            [0.0, 0.0, 0.5],  # mid(0,3)
            [0.5, 0.0, 0.5],  # mid(1,3)
            [0.0, 0.5, 0.5],  # mid(2,3)
        ]
    )
    elements = np.array([list(range(10))])
    mesh = UnstructuredMesh(nodes, elements, "tet10")
    face = PlaneSlab(point=(1.0, 0.0, 0.0), normal=(1.0, 1.0, 1.0), tol=1e-6)
    traction = (0.0, 0.0, 3.0)
    m = UnstructuredElasticity(
        mesh,
        Material(E=1.0, nu=0.3, rho=1.0),
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
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.0, 0.0],  # mid(0,1)
            [0.5, 0.5, 0.0],  # mid(1,2)
            [0.0, 0.5, 0.0],  # mid(2,0)
            [0.0, 0.0, 0.5],  # mid(0,3)
            [0.5, 0.0, 0.5],  # mid(1,3)
            [0.0, 0.5, 0.5],  # mid(2,3)
        ]
    )
    elements = np.array([list(range(10))])
    mesh = UnstructuredMesh(nodes, elements, "tet10")
    volume = 1.0 / 6.0
    b = np.array([2.0, -3.0, 5.0])
    m = UnstructuredElasticity(
        mesh,
        Material(E=1.0, nu=0.3, rho=1.0),
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
# Construction errors: same message family as the grid model
# ---------------------------------------------------------------------------


def test_unsupported_element_kind_rejected() -> None:
    mesh = _tri3_square()

    class FakeKindMesh(UnstructuredMesh):
        @property
        def element_kind(self) -> str:
            return "quad4"

    fake = FakeKindMesh(mesh.nodes, mesh.element_nodes, "tri3")
    with pytest.raises(FemError, match="quad4"):
        UnstructuredElasticity(
            fake,
            Material(E=1.0, nu=0.3, rho=1.0),
            supports=[(NearPoint((0.0, 0.0)), "all")],
            loads=[SurfaceTraction(Box((1.0, 0.0), (1.0, 1.0), tol=1e-9), traction=(1.0, 0.0))],
        )


def test_support_selecting_nothing_raises() -> None:
    mesh = _tri3_square()
    far = Box((50.0, 50.0), (51.0, 51.0), tol=0.0)
    with pytest.raises(FemError, match="no active nodes"):
        UnstructuredElasticity(
            mesh,
            Material(E=1.0, nu=0.3, rho=1.0),
            supports=[(far, "all")],
            loads=[SurfaceTraction(Box((1.0, 0.0), (1.0, 1.0), tol=1e-9), traction=(1.0, 0.0))],
        )


def test_empty_supports_rejected() -> None:
    mesh = _tri3_square()
    with pytest.raises(FemError, match="at least one support"):
        UnstructuredElasticity(
            mesh,
            Material(E=1.0, nu=0.3, rho=1.0),
            supports=[],
            loads=[SurfaceTraction(Box((1.0, 0.0), (1.0, 1.0), tol=1e-9), traction=(1.0, 0.0))],
        )


def test_empty_loads_rejected() -> None:
    mesh = _tri3_square()
    with pytest.raises(FemError, match="at least one load"):
        UnstructuredElasticity(
            mesh,
            Material(E=1.0, nu=0.3, rho=1.0),
            supports=[(NearPoint((0.0, 0.0)), "all")],
            loads=[],
        )


def test_mode_rejected_for_3d() -> None:
    mesh = _tet4_box(1, 1, 1)
    with pytest.raises(FemError, match="2D"):
        _uniaxial_3d_setup(mesh, Material(E=1.0, nu=0.3, rho=1.0), sigma=1.0, mode="plane_strain")


# ---------------------------------------------------------------------------
# Structural sanity: same call surface as the grid model
# ---------------------------------------------------------------------------


def test_satisfies_physics_protocol() -> None:
    mesh = _tri3_square()
    model: PhysicsModel = UnstructuredElasticity(
        mesh,
        Material(E=1.0, nu=0.3, rho=1.0),
        supports=[(NearPoint((0.0, 0.0)), "all")],
        loads=[SurfaceTraction(Box((1.0, 0.0), (1.0, 1.0), tol=1e-9), traction=(1.0, 0.0))],
    )
    assert model.expected_field == FieldSpec("stiffness_scale")
    assert model.n_cases == 1


def test_unstructured_elasticity_registered_as_builtin() -> None:
    from topokit.fem import UnstructuredElasticity as PublicUnstructuredElasticity
    from topokit.registry import registry

    assert registry.get("physics", "unstructured_elasticity") is UnstructuredElasticity
    assert PublicUnstructuredElasticity is UnstructuredElasticity


# ---------------------------------------------------------------------------
# End-to-end integration: FD, convergence, tet10 vs tet4, and Study smoke runs
# ---------------------------------------------------------------------------


def test_compliance_gradient_fd_through_full_stack_tri3() -> None:
    """Mirrors ``test_responses.test_compliance_gradient_fd_through_full_stack``.

    ``DensityFilter`` on a non-grid mesh dispatches to its sparse-matrix path
    (``_BoundMatrixFilter``); this is the first FD check of that path chained
    all the way through ``UnstructuredElasticity`` and ``Direct``.
    """
    mesh = _tri3_grid(4, 3)
    mat = Material(E=100.0, nu=0.3, rho=1.0)
    left = PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0), tol=1e-9)
    tip = Box((4.0, 0.0), (4.0, 3.0), tol=1e-9)
    model = UnstructuredElasticity(
        mesh, mat, supports=[(left, "all")], loads=[SurfaceTraction(tip, traction=(1.0, 0.0))]
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


def _tri3_box(length: float, height: float, nx: int, ny: int) -> UnstructuredMesh:
    """Like ``_tri3_grid`` but over a fixed physical size, decoupled from ``nx``/``ny``.

    Lets a convergence test refine the mesh (``nx``, ``ny`` up) while keeping the
    same domain, mirroring ``StructuredGrid.box``.
    """
    xs, ys = np.linspace(0.0, length, nx + 1), np.linspace(0.0, height, ny + 1)
    grid = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.stack([g.ravel(order="F") for g in grid], axis=1)

    def nid(i: int, j: int) -> int:
        return i + (nx + 1) * j

    elements = []
    for i in range(nx):
        for j in range(ny):
            n0, n1, n2, n3 = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
            elements.append([n0, n1, n2])
            elements.append([n0, n2, n3])
    return UnstructuredMesh(nodes, np.array(elements, dtype=np.int64), "tri3")


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
        mesh = _tri3_box(length, height, nx, ny)
        mat = Material(E=e, nu=nu, rho=1.0)
        left = PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0), tol=1e-9)
        tip = Box((length, 0.0), (length, height), tol=1e-9)
        m = UnstructuredElasticity(
            mesh, mat, supports=[(left, "all")], loads=[PointLoad(tip, (0.0, -p))]
        )
        u = _solve(m, _ones(mesh))
        disp = m.displacement_field(u)
        tip_nodes = np.isclose(mesh.nodes[:, 0], length)
        return float(-disp.values[tip_nodes, 1].min())

    coarse = tip_deflection(24, 4)
    fine = tip_deflection(48, 8)
    assert abs(fine - reference) < abs(coarse - reference)  # converging
    assert abs(fine - reference) / reference < 0.1


def _scaled_tet_mesh(
    nx: int, ny: int, nz: int, size: tuple[float, float, float], kind: str
) -> UnstructuredMesh:
    """A Kuhn ``nx*ny*nz`` box (see the module docstring), rescaled to ``size``.

    Decouples the mesh's element-count resolution from its physical extent,
    so a tet4 and a tet10 mesh at different (nx, ny, nz) can still cover the
    identical domain -- needed to compare them at equal DOF on one cantilever.
    """
    nodes4, elements4 = _tet4_box_arrays(nx, ny, nz)
    nodes, elements = (nodes4, elements4) if kind == "tet4" else _tet4_to_tet10(nodes4, elements4)
    scale = np.array([size[0] / nx, size[1] / ny, size[2] / nz])
    return UnstructuredMesh(nodes * scale, elements, kind)


def _solve_any(model: Any, scale: np.ndarray) -> np.ndarray:
    """Like ``_solve``, generalized over ``LinearElasticity``/``UnstructuredElasticity``."""
    solver = Direct()
    solver.prepare(model.assemble(scale))
    return np.asarray(solver.solve(model.loads()[:, 0]))


def test_tet10_closer_than_tet4_at_equal_dof_budget() -> None:
    """tet10 (quadratic) beats tet4 (linear) at matched DOF on one cantilever.

    Both meshes cover the same 6x2x2 box (see ``_scaled_tet_mesh``): tet4 is
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
    left = PlaneSlab(point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), tol=1e-9)
    tip_sel = Box((length, 0.0, 0.0), (length, side, side), tol=1e-9)
    tip_load = [PointLoad(tip_sel, (0.0, 0.0, -p))]

    def tip_deflection(model: Any) -> float:
        u = _solve_any(model, np.ones(model.mesh.n_elements))
        disp = model.displacement_field(u)
        tip_nodes = np.isclose(model.mesh.nodes[:, 0], length)
        return float(-disp.values[tip_nodes, 2].min())

    g = StructuredGrid.box(size=size, shape=(24, 8, 8))
    hex_model = LinearElasticity(g, mat, supports=[(left, "all")], loads=tip_load)
    reference = tip_deflection(hex_model)

    tet4_mesh = _scaled_tet_mesh(13, 2, 2, size, "tet4")
    tet10_mesh = _scaled_tet_mesh(2, 2, 2, size, "tet10")
    assert tet4_mesh.n_nodes * 3 == pytest.approx(tet10_mesh.n_nodes * 3, rel=0.02)  # equal DOF

    tet4_model = UnstructuredElasticity(tet4_mesh, mat, supports=[(left, "all")], loads=tip_load)
    tet10_model = UnstructuredElasticity(tet10_mesh, mat, supports=[(left, "all")], loads=tip_load)
    tet4_tip = tip_deflection(tet4_model)
    tet10_tip = tip_deflection(tet10_model)

    assert abs(tet10_tip - reference) < abs(tet4_tip - reference)


def test_study_smoke_tri3_oc() -> None:
    """Full ``Study`` smoke run on a tri3 cantilever built from arrays in-test."""
    nx, ny = 8, 4
    xs, ys = np.arange(nx + 1), np.arange(ny + 1)
    grid = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.stack([g.ravel(order="F") for g in grid], axis=1).astype(float)

    def nid(i: int, j: int) -> int:
        return i + (nx + 1) * j

    elements = []
    for i in range(nx):
        for j in range(ny):
            n0, n1, n2, n3 = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
            elements.append([n0, n1, n2])
            elements.append([n0, n2, n3])
    mesh = UnstructuredMesh(nodes, np.array(elements, dtype=np.int64), "tri3")

    mat = Material(E=1.0, nu=0.3, rho=1.0)
    left = PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0), tol=1e-9)
    tip = Box((float(nx), 0.0), (float(nx), float(ny)), tol=1e-9)
    model = UnstructuredElasticity(
        mesh, mat, supports=[(left, "all")], loads=[PointLoad(tip, (0.0, -1.0))]
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
    mesh = _tet4_box(2, 2, 2)
    mat = Material(E=1.0, nu=0.3, rho=1.0)
    left = PlaneSlab(point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), tol=1e-9)
    tip = Box((2.0, 0.0, 0.0), (2.0, 2.0, 2.0), tol=1e-9)
    model = UnstructuredElasticity(
        mesh, mat, supports=[(left, "all")], loads=[PointLoad(tip, (0.0, 0.0, -1.0))]
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
