# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Conformance suite the Mesh protocol requires of every implementation.

Runs identical checks against `StructuredGrid` and `UnstructuredMesh` so a
regression in either implementation shows up immediately. If an assertion
here fails against `StructuredGrid`, the assertion is wrong: `StructuredGrid`
is the reference implementation, not the suite.

The protocol members `MeshBase` derives (`active_elements`, `active_nodes`,
`node_index_map`, the mask validation and defaults, the `boundary_faces`
cache) are checked once against a minimal subclass at the bottom of this
file instead of once per fixture: both shipped implementations inherit the
one implementation, so a second copy of the check pins nothing extra.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from topokit.mesh import (
    BoundaryFaces,
    Mesh,
    MeshBase,
    MeshError,
    StructuredGrid,
    UnstructuredMesh,
)


@dataclass(frozen=True)
class Fixture:
    """A `Mesh` instance plus the known geometric measures to check it against."""

    id: str
    mesh: Mesh
    total_volume: float
    boundary_measure: float


def _structured_2d() -> Fixture:
    # elements are x-fastest: solid at x in [0,1], design at x in [1,2],
    # void at x in [2,3]; the active region is the 2x1 rectangle [0,2]x[0,1]
    g = StructuredGrid(
        shape=(3, 1),
        spacing=(1.0, 1.0),
        solid=[True, False, False],
        void=[False, False, True],
    )
    return Fixture("structured_2d", g, total_volume=3.0, boundary_measure=6.0)


def _structured_3d() -> Fixture:
    # layers along z: solid at z in [0,1], design at z in [1,2], void at
    # z in [2,3]; the active region is the 1x1x2 box [0,1]x[0,1]x[0,2]
    g = StructuredGrid(
        shape=(1, 1, 3),
        spacing=(1.0, 1.0, 1.0),
        solid=[True, False, False],
        void=[False, False, True],
    )
    return Fixture("structured_3d", g, total_volume=3.0, boundary_measure=10.0)


def _tri3_square() -> Fixture:
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    elements = np.array([[0, 1, 2], [0, 2, 3]])
    m = UnstructuredMesh(nodes, elements, "tri3")
    return Fixture("tri3_square", m, total_volume=1.0, boundary_measure=4.0)


def _tet4_cube_slice() -> Fixture:
    # one third of the unit cube: a square-base pyramid with apex above one
    # base corner, split along the base diagonal into two tet4s
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    elements = np.array([[0, 1, 2, 4], [0, 2, 3, 4]])
    m = UnstructuredMesh(nodes, elements, "tet4")
    return Fixture(
        "tet4_cube_slice", m, total_volume=1.0 / 3.0, boundary_measure=2.0 + math.sqrt(2.0)
    )


def _tet10_cube_slice() -> Fixture:
    # midside-node (VTK_QUADRATIC_TETRA) upgrade of _tet4_cube_slice(): same
    # straight-edge geometry, so volume/surface (both corner-based) match
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            [0.0, 0.0, 1.0],  # 4
            [0.5, 0.0, 0.0],  # 5  mid(0,1)
            [1.0, 0.5, 0.0],  # 6  mid(1,2)
            [0.5, 0.5, 0.0],  # 7  mid(2,0)
            [0.0, 0.0, 0.5],  # 8  mid(0,4)
            [0.5, 0.0, 0.5],  # 9  mid(1,4)
            [0.5, 0.5, 0.5],  # 10 mid(2,4)
            [0.5, 1.0, 0.0],  # 11 mid(2,3)
            [0.0, 0.5, 0.0],  # 12 mid(3,0)
            [0.0, 0.5, 0.5],  # 13 mid(3,4)
        ]
    )
    elements = np.array(
        [
            [0, 1, 2, 4, 5, 6, 7, 8, 9, 10],
            [0, 2, 3, 4, 7, 11, 12, 8, 10, 13],
        ]
    )
    m = UnstructuredMesh(nodes, elements, "tet10")
    return Fixture(
        "tet10_cube_slice", m, total_volume=1.0 / 3.0, boundary_measure=2.0 + math.sqrt(2.0)
    )


def _tri3_void() -> Fixture:
    # two unit squares, each split into 2 triangles; the right square's pair
    # is void, so the active domain is the unit square [0,1]x[0,1] (perimeter
    # 4) and the two x=2 nodes become inactive
    nodes = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ]
    )
    elements = np.array([[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]])
    m = UnstructuredMesh(
        nodes, elements, "tri3", solid=[True, False, False, False], void=[False, False, True, True]
    )
    return Fixture("tri3_void", m, total_volume=2.0, boundary_measure=4.0)


def _tet10_void() -> Fixture:
    # _tet10_cube_slice() with its second tet voided: the active domain is the
    # single tet (0,1,2,4), all 4 of whose faces are boundary, and corner 3
    # plus the 3 midsides only it touches become inactive
    base = _tet10_cube_slice().mesh
    m = UnstructuredMesh(
        base.nodes,
        base.element_nodes,
        "tet10",
        void=[False, True],
    )
    return Fixture("tet10_void", m, total_volume=1.0 / 3.0, boundary_measure=1.0 + math.sqrt(2.0))


FIXTURES = [
    _structured_2d(),
    _structured_3d(),
    _tri3_square(),
    _tri3_void(),
    _tet4_cube_slice(),
    _tet10_cube_slice(),
    _tet10_void(),
]


@pytest.mark.parametrize("fx", FIXTURES, ids=[f.id for f in FIXTURES])
class TestMeshConformance:
    """Every `Mesh` implementation must satisfy these checks identically."""

    def test_protocol_member_shapes_and_dtypes(self, fx: Fixture) -> None:
        m = fx.mesh
        assert isinstance(m, Mesh)
        n_nodes, n_el, dim = m.n_nodes, m.n_elements, m.dim
        assert m.nodes.shape == (n_nodes, dim)
        assert m.nodes.dtype == np.float64
        assert m.element_nodes.shape[0] == n_el
        assert m.element_nodes.dtype == np.int64
        assert m.element_volumes.shape == (n_el,)
        assert m.element_volumes.dtype == np.float64
        assert m.element_centroids.shape == (n_el, dim)
        assert m.element_centroids.dtype == np.float64
        for mask in (m.design, m.solid, m.void, m.active_elements):
            assert mask.shape == (n_el,)
            assert mask.dtype == np.bool_
        assert m.active_nodes.shape == (n_nodes,)
        assert m.active_nodes.dtype == np.bool_
        assert m.node_index_map.shape == (n_nodes,)
        assert m.node_index_map.dtype == np.int64

    def test_masks_disjoint_and_not_all_void(self, fx: Fixture) -> None:
        m = fx.mesh
        assert not (m.solid & m.void).any()
        assert not m.void.all()

    def test_element_volumes_positive_and_sum_to_domain_measure(self, fx: Fixture) -> None:
        m = fx.mesh
        assert (m.element_volumes > 0).all()
        assert m.element_volumes.sum() == pytest.approx(fx.total_volume)

    def test_centroids_inside_element_bounding_box(self, fx: Fixture) -> None:
        m = fx.mesh
        for e in range(m.n_elements):
            corners = m.nodes[m.element_nodes[e]]
            lo, hi = corners.min(axis=0), corners.max(axis=0)
            centroid = m.element_centroids[e]
            assert (centroid >= lo - 1e-12).all()
            assert (centroid <= hi + 1e-12).all()

    def test_boundary_faces_owners_are_active(self, fx: Fixture) -> None:
        m = fx.mesh
        f = m.boundary_faces()
        assert m.active_elements[f.owner].all()

    def test_boundary_faces_normals_are_unit_and_outward(self, fx: Fixture) -> None:
        m = fx.mesh
        f = m.boundary_faces()
        np.testing.assert_allclose(np.linalg.norm(f.normal, axis=1), 1.0)
        owner_centroid = m.element_centroids[f.owner]
        outward = np.einsum("ij,ij->i", f.normal, f.centroid - owner_centroid)
        assert (outward > 0.0).all()

    def test_boundary_faces_total_area_matches_known_surface(self, fx: Fixture) -> None:
        m = fx.mesh
        f = m.boundary_faces()
        assert f.area.sum() == pytest.approx(fx.boundary_measure)

    def test_derived_arrays_are_write_protected(self, fx: Fixture) -> None:
        m = fx.mesh
        f = m.boundary_faces()
        for arr in (
            m.nodes,
            m.element_nodes,
            m.element_volumes,
            m.element_centroids,
            m.design,
            m.solid,
            m.void,
            m.active_elements,
            m.active_nodes,
            m.node_index_map,
            f.owner,
            f.normal,
            f.area,
            f.centroid,
            f.nodes,
        ):
            with pytest.raises(ValueError, match="read-only"):
                arr[0] = 0


# --- MeshBase: the derived members, checked once ----------------------


class _TwoTriangles(MeshBase):
    # The whole authoring surface of MeshBase, and nothing else: the five
    # geometry members, the masks, and the face builder. If this class needs
    # to grow, the base stopped deriving something it used to.
    def __init__(self, solid: list[bool] | None = None, void: list[bool] | None = None) -> None:
        self._nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        self._elements = np.array([[0, 1, 2], [0, 2, 3]])
        self.face_builds = 0
        self._set_masks(solid, void)

    @property
    def element_kind(self) -> str:
        return "tri3"

    @property
    def nodes(self) -> np.ndarray:
        return self._nodes

    @property
    def element_nodes(self) -> np.ndarray:
        return self._elements

    @property
    def element_volumes(self) -> np.ndarray:
        return np.full(2, 0.5)

    @property
    def element_centroids(self) -> np.ndarray:
        return np.asarray(self._nodes[self._elements].mean(axis=1))

    def _compute_boundary_faces(self) -> BoundaryFaces:
        self.face_builds += 1
        return BoundaryFaces(
            owner=np.array([0, 0, 1, 1]),
            normal=np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
            area=np.ones(4),
            centroid=np.array([[0.5, 0.0], [1.0, 0.5], [0.5, 1.0], [0.0, 0.5]]),
            nodes=np.array([[0, 1], [1, 2], [2, 3], [3, 0]]),
        )


def test_base_fills_in_the_rest_of_the_protocol() -> None:
    assert isinstance(_TwoTriangles(), Mesh)


def test_base_counts_come_off_the_arrays() -> None:
    m = _TwoTriangles()
    assert (m.n_nodes, m.n_elements, m.dim) == (4, 2, 2)


def test_base_masks_default_to_all_design() -> None:
    m = _TwoTriangles()
    assert m.design.all()
    assert not m.solid.any()
    assert not m.void.any()


def test_base_active_elements_is_not_void() -> None:
    m = _TwoTriangles(void=[False, True])
    np.testing.assert_array_equal(m.active_elements, ~m.void)
    np.testing.assert_array_equal(m.design, [True, False])


def test_base_active_nodes_match_active_element_references() -> None:
    m = _TwoTriangles(void=[False, True])
    referenced = set(m.element_nodes[m.active_elements].ravel().tolist())
    assert set(np.nonzero(m.active_nodes)[0].tolist()) == referenced


def test_base_node_index_map_is_consecutive_over_active_nodes() -> None:
    m = _TwoTriangles(void=[False, True])
    n_active = int(m.active_nodes.sum())
    np.testing.assert_array_equal(m.node_index_map[m.active_nodes], np.arange(n_active))
    assert (m.node_index_map[~m.active_nodes] == -1).all()


def test_base_builds_boundary_faces_once_and_caches_them() -> None:
    m = _TwoTriangles()
    assert m.boundary_faces() is m.boundary_faces()
    assert m.face_builds == 1


def test_base_validates_the_masks_it_is_given() -> None:
    with pytest.raises(MeshError, match="shape"):
        _TwoTriangles(void=[True])
    with pytest.raises(MeshError, match="all elements are void"):
        _TwoTriangles(void=[True, True])
    with pytest.raises(MeshError, match="disjoint"):
        _TwoTriangles(solid=[True, False], void=[True, False])
