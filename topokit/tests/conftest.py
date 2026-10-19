# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Values and builders the suite had written out again in file after file.

The unit material appeared 29 times, the fixed-left-edge selector 43, the
corner tet10 coordinates 9 and the unit-square tri3 8. They are plain
module-level names rather than fixtures because `pytest.mark.parametrize`
runs at collection time and cannot take a fixture -- the two conformance
suites build their parameter lists from these -- and because a name in the
test body reads better than one more argument in its signature. Import what
you need: ``from tests.conftest import UNIT_MATERIAL, solve``.

Nothing here hides what a test is about: each entry is one named value or
one small mesh builder. Anything with a load, a support or an assertion in
it stays in the test file that cares.
"""

from __future__ import annotations

import numpy as np

from topokit.backend import NumpyBackend
from topokit.fem import Material, PhysicsModel
from topokit.mesh import Mesh, UnstructuredMesh
from topokit.selection import PlaneSlab
from topokit.solvers import Direct

# E=1 keeps displacements O(1) and compliance readable; nu=0.3 is the steel
# value every patch and energy test was written against.
UNIT_MATERIAL = Material(E=1.0, nu=0.3, rho=1.0)

# The cantilever root: the x=0 plane, exactly. tol=1e-9 rather than
# `PlaneSlab`'s element-scale default, which would reach half an element in.
LEFT_EDGE_2D = PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0), tol=1e-9)
LEFT_EDGE_3D = PlaneSlab(point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), tol=1e-9)

# A single corner tet10: straight edges, midsides at the exact edge midpoints,
# node order VTK_QUADRATIC_TETRA (4 corners, then the midsides of edges
# (0,1),(1,2),(2,0),(0,3),(1,3),(2,3)). Physical coordinates equal the
# reference (r, s, t), which is what keeps the analytic derivations in
# test_elements.py tractable.
CORNER_TET10_NODES = np.array(
    [
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [0.0, 1.0, 0.0],  # 2
        [0.0, 0.0, 1.0],  # 3
        [0.5, 0.0, 0.0],  # 4 mid(0,1)
        [0.5, 0.5, 0.0],  # 5 mid(1,2)
        [0.0, 0.5, 0.0],  # 6 mid(2,0)
        [0.0, 0.0, 0.5],  # 7 mid(0,3)
        [0.5, 0.0, 0.5],  # 8 mid(1,3)
        [0.0, 0.5, 0.5],  # 9 mid(2,3)
    ]
)
CORNER_TET10_ELEMENTS = np.array([list(range(10))], dtype=np.int64)

# The unit square, split along the (0,2) diagonal into two tri3s.
SQUARE_NODES = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
SQUARE_TRI3 = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

_TET10_MIDSIDE_PAIRS = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))

# Kuhn (Freudenthal) decomposition of a cube into 6 tets, corners only: for
# each of the 3! axis orderings, walk the cube diagonal one axis at a time.
# Local corner order is (0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,0,1),(1,0,1),
# (0,1,1),(1,1,1); the element rows are reordered from the raw permutation
# walk so every tet has positive signed volume (`UnstructuredMesh` requires it).
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


def _lattice(counts: tuple[int, ...], size: tuple[float, ...] | None) -> np.ndarray:
    """x-fastest node lattice over ``size`` (unit cells when ``size`` is None)."""
    extent = tuple(float(n) for n in counts) if size is None else size
    axes = [np.linspace(0.0, s, n + 1) for s, n in zip(extent, counts, strict=True)]
    grid = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel(order="F") for g in grid], axis=1)


def tri3_square() -> UnstructuredMesh:
    """The 2-triangle unit square."""
    return UnstructuredMesh(SQUARE_NODES, SQUARE_TRI3, "tri3")


def corner_tet10() -> UnstructuredMesh:
    """The single corner tet10."""
    return UnstructuredMesh(CORNER_TET10_NODES, CORNER_TET10_ELEMENTS, "tet10")


def tri3_grid(nx: int, ny: int, size: tuple[float, float] | None = None) -> UnstructuredMesh:
    """An ``nx*ny`` quad lattice, each cell split into 2 tri3s on one diagonal.

    Elements run cell-major (x outer, y inner), 2 per cell, so cell ``(i, j)``
    owns elements ``2*(i*ny + j)`` and ``+1``.
    """
    nodes = _lattice((nx, ny), size)
    elements = []
    for i in range(nx):
        for j in range(ny):
            n0 = i + (nx + 1) * j
            n1, n2, n3 = n0 + 1, n0 + 1 + (nx + 1), n0 + (nx + 1)
            elements += [[n0, n1, n2], [n0, n2, n3]]
    return UnstructuredMesh(nodes, np.array(elements, dtype=np.int64), "tri3")


def _tet4_arrays(
    nx: int, ny: int, nz: int, size: tuple[float, float, float] | None
) -> tuple[np.ndarray, np.ndarray]:
    nodes = _lattice((nx, ny, nz), size)

    def nid(i: int, j: int, k: int) -> int:
        return i + (nx + 1) * j + (nx + 1) * (ny + 1) * k

    elements = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                corners = [nid(i + di, j + dj, k + dk) for di, dj, dk in _KUHN_CUBE_LOCAL]
                elements += [[corners[t] for t in tet] for tet in _KUHN_CUBE_TETS]
    return nodes, np.array(elements, dtype=np.int64)


def tet4_box(
    nx: int, ny: int, nz: int, size: tuple[float, float, float] | None = None
) -> UnstructuredMesh:
    """A Kuhn triangulation of an ``nx*ny*nz`` cube lattice into 6 tet4s each.

    Corner nodes only, shared between adjacent cubes: a conforming tet4 mesh
    with no extra nodes, built by hand so the suite needs no mesh generator.
    """
    return UnstructuredMesh(*_tet4_arrays(nx, ny, nz, size), "tet4")


def tet10_box(
    nx: int, ny: int, nz: int, size: tuple[float, float, float] | None = None
) -> UnstructuredMesh:
    """`tet4_box` upgraded to tet10: one midpoint node per unique edge, shared."""
    nodes, elements = _tet4_arrays(nx, ny, nz, size)
    node_list = list(nodes)
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
    return UnstructuredMesh(np.array(node_list), np.array(rows, dtype=np.int64), "tet10")


def ones(mesh: Mesh) -> np.ndarray:
    """Unit stiffness scale over every element of ``mesh``."""
    return np.ones(mesh.n_elements)


def solve(model: PhysicsModel, scale: np.ndarray | None = None) -> np.ndarray:
    """Direct solve of ``model``'s first load case at ``scale`` (unit by default)."""
    solver = Direct()
    solver.prepare(model.assemble(ones(model.mesh) if scale is None else scale))
    return np.asarray(solver.solve(model.loads()[:, 0]))


def named_backend(name: str) -> NumpyBackend:
    """A `NumpyBackend` reporting ``name``, for the kernel-dispatch tests."""

    class Named(NumpyBackend):
        @property
        def name(self) -> str:
            return name

    return Named()
