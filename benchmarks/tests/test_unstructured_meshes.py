# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
from __future__ import annotations

import numpy as np
import pytest

from topokit_bench.problems import _tet10_mesh, _tet_mesh, _tri_mesh

_TET_LOCAL_EDGES: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))


def test_tri_mesh_construction_and_measures() -> None:
    nelx, nely = 3, 2
    sx, sy = 6.0, 3.0
    mesh = _tri_mesh(nelx, nely, (sx, sy))
    assert mesh.element_kind == "tri3"
    assert mesh.n_elements == 2 * nelx * nely
    assert mesh.n_nodes == (nelx + 1) * (nely + 1)
    assert mesh.element_volumes.sum() == pytest.approx(sx * sy, abs=1e-12)
    faces = mesh.boundary_faces()
    assert faces.area.sum() == pytest.approx(2.0 * (sx + sy), abs=1e-12)


def test_tri_mesh_deterministic() -> None:
    a = _tri_mesh(3, 2, (6.0, 3.0))
    b = _tri_mesh(3, 2, (6.0, 3.0))
    np.testing.assert_array_equal(a.nodes, b.nodes)
    np.testing.assert_array_equal(a.element_nodes, b.element_nodes)


def test_tet_mesh_construction_and_measures() -> None:
    # Asymmetric element counts and box size per axis: a bug that mixes up
    # x/y/z (e.g. using nx1 where ny1 belongs) would corrupt connectivity or
    # the volume/surface sums here, unlike a nelx==nely==nelz cube.
    nelx, nely, nelz = 3, 2, 4
    sx, sy, sz = 3.0, 2.0, 4.0
    mesh = _tet_mesh(nelx, nely, nelz, (sx, sy, sz))
    assert mesh.element_kind == "tet4"
    assert mesh.n_elements == 6 * nelx * nely * nelz
    assert mesh.n_nodes == (nelx + 1) * (nely + 1) * (nelz + 1)
    assert mesh.element_volumes.sum() == pytest.approx(sx * sy * sz, abs=1e-12)
    faces = mesh.boundary_faces()
    surface = 2.0 * (sx * sy + sy * sz + sz * sx)
    assert faces.area.sum() == pytest.approx(surface, abs=1e-12)


def test_tet_mesh_deterministic() -> None:
    a = _tet_mesh(3, 2, 4, (3.0, 2.0, 4.0))
    b = _tet_mesh(3, 2, 4, (3.0, 2.0, 4.0))
    np.testing.assert_array_equal(a.nodes, b.nodes)
    np.testing.assert_array_equal(a.element_nodes, b.element_nodes)


def test_tet10_mesh_construction_and_measures() -> None:
    nelx, nely, nelz = 3, 2, 4
    size = (3.0, 2.0, 4.0)
    tet4 = _tet_mesh(nelx, nely, nelz, size)
    mesh = _tet10_mesh(nelx, nely, nelz, size)
    assert mesh.element_kind == "tet10"
    assert mesh.n_elements == tet4.n_elements

    # Closed-form node count: lattice nodes + one midpoint per unique tet4
    # edge, the unique-edge set derived here (not a hand constant).
    edges = {
        (int(row[a]), int(row[b])) if row[a] < row[b] else (int(row[b]), int(row[a]))
        for row in tet4.element_nodes
        for a, b in _TET_LOCAL_EDGES
    }
    assert mesh.n_nodes == tet4.n_nodes + len(edges)

    assert mesh.element_volumes.sum() == pytest.approx(size[0] * size[1] * size[2], abs=1e-12)
    faces = mesh.boundary_faces()
    surface = 2.0 * (size[0] * size[1] + size[1] * size[2] + size[2] * size[0])
    assert faces.area.sum() == pytest.approx(surface, abs=1e-12)


def test_tet10_mesh_deterministic() -> None:
    a = _tet10_mesh(3, 2, 4, (3.0, 2.0, 4.0))
    b = _tet10_mesh(3, 2, 4, (3.0, 2.0, 4.0))
    np.testing.assert_array_equal(a.nodes, b.nodes)
    np.testing.assert_array_equal(a.element_nodes, b.element_nodes)
