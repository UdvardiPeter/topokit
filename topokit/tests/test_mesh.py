# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Tests for the structured grid and the unstructured simplex mesh."""

import math

import numpy as np
import pytest

from topokit.mesh import Mesh, MeshError, StructuredGrid, UnstructuredMesh


def test_box_2d_counts() -> None:
    g = StructuredGrid.box(size=(2.0, 1.0), element_size=0.5)
    assert g.shape == (4, 2)
    assert g.n_elements == 8
    assert g.n_nodes == 15
    assert g.dim == 2
    assert g.element_kind == "quad4"


def test_box_3d_counts() -> None:
    g = StructuredGrid.box(size=(1.0, 1.0, 1.0), element_size=0.5)
    assert g.shape == (2, 2, 2)
    assert g.n_elements == 8
    assert g.n_nodes == 27
    assert g.element_kind == "hex8"


def test_box_anisotropic_spacing() -> None:
    g = StructuredGrid.box(size=(2.0, 1.0), element_size=(0.5, 0.25))
    assert g.shape == (4, 4)
    np.testing.assert_allclose(g.spacing, (0.5, 0.25))


def test_box_recomputes_spacing_to_cover_size_exactly() -> None:
    g = StructuredGrid.box(size=(1.0, 1.0), element_size=0.3)
    assert g.shape == (3, 3)
    np.testing.assert_allclose(g.spacing, (1.0 / 3.0, 1.0 / 3.0))


def test_box_from_shape() -> None:
    g = StructuredGrid.box(size=(2.0, 1.0), shape=(8, 4))
    np.testing.assert_allclose(g.spacing, (0.25, 0.25))


def test_node_ordering_x_fastest() -> None:
    g = StructuredGrid(shape=(2, 2), spacing=(0.5, 0.25))
    np.testing.assert_allclose(g.nodes[0], [0.0, 0.0])
    np.testing.assert_allclose(g.nodes[1], [0.5, 0.0])
    np.testing.assert_allclose(g.nodes[3], [0.0, 0.25])
    np.testing.assert_allclose(g.nodes[8], [1.0, 0.5])


def test_quad4_connectivity_vtk_order() -> None:
    g = StructuredGrid(shape=(2, 2), spacing=(1.0, 1.0))
    np.testing.assert_array_equal(g.element_nodes[0], [0, 1, 4, 3])
    np.testing.assert_array_equal(g.element_nodes[3], [4, 5, 8, 7])


def test_hex8_connectivity_vtk_order() -> None:
    g = StructuredGrid(shape=(2, 1, 1), spacing=(1.0, 1.0, 1.0))
    np.testing.assert_array_equal(g.element_nodes[1], [1, 2, 5, 4, 7, 8, 11, 10])


def test_element_centroids_and_volumes() -> None:
    g = StructuredGrid(shape=(2, 1), spacing=(0.5, 0.25), origin=(1.0, 2.0))
    np.testing.assert_allclose(g.element_centroids[0], [1.25, 2.125])
    np.testing.assert_allclose(g.element_centroids[1], [1.75, 2.125])
    np.testing.assert_allclose(g.element_volumes, [0.125, 0.125])


def test_construction_validation() -> None:
    with pytest.raises(MeshError, match="dim"):
        StructuredGrid(shape=(4,), spacing=(1.0,))
    with pytest.raises(MeshError, match="dim"):
        StructuredGrid(shape=(2, 2, 2, 2), spacing=(1.0, 1.0, 1.0, 1.0))
    with pytest.raises(MeshError, match="spacing"):
        StructuredGrid(shape=(2, 2), spacing=(1.0, -1.0))
    with pytest.raises(MeshError, match="length"):
        StructuredGrid(shape=(2, 2), spacing=(1.0, 1.0, 1.0))


def test_masks_default_all_design() -> None:
    g = StructuredGrid(shape=(2, 2), spacing=(1.0, 1.0))
    assert g.design.all()
    assert not g.solid.any()
    assert not g.void.any()


def test_masks_partition() -> None:
    solid = np.array([True, False, False, False])
    void = np.array([False, False, False, True])
    g = StructuredGrid(shape=(2, 2), spacing=(1.0, 1.0), solid=solid, void=void)
    np.testing.assert_array_equal(g.design, [False, True, True, False])
    assert not (g.design & g.solid).any()
    assert (g.design | g.solid | g.void).all()


def test_overlapping_masks_raise() -> None:
    both = np.array([True, False, False, False])
    with pytest.raises(MeshError, match="disjoint"):
        StructuredGrid(shape=(2, 2), spacing=(1.0, 1.0), solid=both, void=both)


def test_wrong_mask_shape_raises() -> None:
    with pytest.raises(MeshError, match="shape"):
        StructuredGrid(shape=(2, 2), spacing=(1.0, 1.0), void=np.array([True]))


def test_all_void_raises() -> None:
    with pytest.raises(MeshError, match="void"):
        StructuredGrid(shape=(2, 1), spacing=(1.0, 1.0), void=np.array([True, True]))


def test_active_nodes_exclude_void_only_nodes() -> None:
    g = StructuredGrid(shape=(2, 1), spacing=(1.0, 1.0), void=np.array([False, True]))
    np.testing.assert_array_equal(g.active_elements, [True, False])
    np.testing.assert_array_equal(g.active_nodes, [True, True, False, True, True, False])
    np.testing.assert_array_equal(g.node_index_map, [0, 1, -1, 2, 3, -1])


def test_boundary_faces_2d_anisotropic() -> None:
    g = StructuredGrid(shape=(1, 1), spacing=(0.5, 0.25))
    f = g.boundary_faces()
    assert f.n_faces == 4
    np.testing.assert_array_equal(f.owner, [0, 0, 0, 0])
    totals = sorted(f.area)
    np.testing.assert_allclose(totals, [0.25, 0.25, 0.5, 0.5])
    norms = np.linalg.norm(f.normal, axis=1)
    np.testing.assert_allclose(norms, 1.0)
    assert f.normal.sum() == pytest.approx(0.0)


def test_boundary_faces_3d_areas() -> None:
    g = StructuredGrid(shape=(1, 1, 1), spacing=(2.0, 3.0, 5.0))
    f = g.boundary_faces()
    assert f.n_faces == 6
    np.testing.assert_allclose(sorted(f.area), [6.0, 6.0, 10.0, 10.0, 15.0, 15.0])
    for k in range(6):
        axis = int(np.argmax(np.abs(f.normal[k])))
        plane = f.centroid[k][axis]
        node_coords = g.nodes[f.nodes[k]]
        np.testing.assert_allclose(node_coords[:, axis], plane)


def test_boundary_faces_include_void_interface() -> None:
    g = StructuredGrid(shape=(2, 1), spacing=(1.0, 1.0), void=np.array([False, True]))
    f = g.boundary_faces()
    assert f.n_faces == 4
    np.testing.assert_array_equal(f.owner, [0, 0, 0, 0])
    plus_x = np.where(f.normal[:, 0] > 0.5)[0]
    assert len(plus_x) == 1
    k = plus_x[0]
    np.testing.assert_allclose(f.centroid[k], [1.0, 0.5])
    assert set(f.nodes[k].tolist()) == {1, 4}


def test_satisfies_mesh_protocol_and_meshlike() -> None:
    from topokit.fields import DesignField

    g = StructuredGrid(shape=(2, 2), spacing=(1.0, 1.0))
    mesh: Mesh = g
    assert mesh.n_elements == 4
    rho = DesignField(np.full(4, 0.5), g, name="rho")
    assert rho.values.shape == (4,)


def test_grid_is_immutable() -> None:
    g = StructuredGrid(shape=(2, 2), spacing=(1.0, 1.0))
    with pytest.raises(AttributeError):
        g.solid = np.ones(4, dtype=bool)  # type: ignore[misc]
    with pytest.raises(AttributeError):
        g.spacing = (2.0, 2.0)  # type: ignore[misc]


def test_derived_arrays_are_read_only() -> None:
    g = StructuredGrid(shape=(2, 2), spacing=(1.0, 1.0))
    for arr in (
        g.nodes,
        g.element_nodes,
        g.element_centroids,
        g.element_volumes,
        g.design,
        g.active_elements,
        g.boundary_faces().area,
        g.boundary_faces().nodes,
    ):
        with pytest.raises(ValueError, match="read-only"):
            arr[0] = 0


def test_boundary_faces_cached() -> None:
    g = StructuredGrid(shape=(2, 2), spacing=(1.0, 1.0))
    assert g.boundary_faces() is g.boundary_faces()


def test_face_nodes_are_perimeter_cyclic() -> None:
    g = StructuredGrid(shape=(2, 2, 2), spacing=(1.0, 2.0, 3.0))
    f = g.boundary_faces()
    assert f.n_faces == 24
    for k in range(f.n_faces):
        coords = g.nodes[f.nodes[k]]
        for a in range(4):
            step = np.abs(coords[(a + 1) % 4] - coords[a])
            assert np.count_nonzero(step) == 1


def test_boundary_faces_3d_void_interface() -> None:
    g = StructuredGrid(shape=(1, 1, 2), spacing=(1.0, 1.0, 1.0), void=np.array([False, True]))
    f = g.boundary_faces()
    assert f.n_faces == 6
    np.testing.assert_array_equal(f.owner, np.zeros(6, dtype=np.int64))
    plus_z = np.where(f.normal[:, 2] > 0.5)[0]
    assert len(plus_z) == 1
    np.testing.assert_allclose(f.centroid[plus_z[0]], [0.5, 0.5, 1.0])


def test_box_accepts_numpy_scalar_element_size() -> None:
    g = StructuredGrid.box(size=(1.0, 1.0), element_size=np.float32(0.5))
    assert g.shape == (2, 2)


def test_pickle_is_lean_and_preserves_immutability() -> None:
    import pickle

    g = StructuredGrid(shape=(20, 20, 20), spacing=(1.0, 1.0, 1.0))
    _ = g.element_nodes  # populate caches
    _ = g.boundary_faces()
    blob = pickle.dumps(g)
    assert len(blob) < 100_000  # caches are not serialized
    g2 = pickle.loads(blob)
    assert not g2.void.flags.writeable
    assert not g2.solid.flags.writeable
    np.testing.assert_array_equal(g2.element_nodes, g.element_nodes)
    assert g2.boundary_faces().n_faces == g.boundary_faces().n_faces


def test_constructor_accepts_plain_lists_as_masks() -> None:
    g = StructuredGrid(shape=(2, 1), spacing=(1.0, 1.0), void=[False, True])
    np.testing.assert_array_equal(g.void, [False, True])


def test_repr_is_informative() -> None:
    g = StructuredGrid(shape=(2, 2), spacing=(1.0, 0.5), void=[False, False, False, True])
    r = repr(g)
    assert "shape=(2, 2)" in r
    assert "1 void" in r


def test_to_grid_to_flat_round_trip_and_layout() -> None:
    g = StructuredGrid(shape=(2, 2), spacing=(1.0, 1.0))
    flat = np.arange(4, dtype=float)
    a = g.to_grid(flat)
    assert a[1, 0] == 1.0  # element id i + nx*j
    assert a[0, 1] == 2.0
    np.testing.assert_array_equal(g.to_flat(a), flat)
    assert np.shares_memory(a, flat)


def test_to_grid_to_flat_validate_shapes() -> None:
    g = StructuredGrid(shape=(2, 2), spacing=(1.0, 1.0))
    with pytest.raises(MeshError, match="flat shape"):
        g.to_grid(np.zeros(5))
    with pytest.raises(MeshError, match="grid shape"):
        g.to_flat(np.zeros((3, 2)))


def test_protocol_exposes_masks_and_condensation() -> None:
    g = StructuredGrid(shape=(2, 1), spacing=(1.0, 1.0), void=[False, True])
    mesh: Mesh = g
    assert mesh.design.sum() == 1
    assert not mesh.solid.any()
    assert mesh.void.sum() == 1
    assert mesh.active_elements.sum() == 1
    assert mesh.active_nodes.sum() == 4
    assert mesh.node_index_map.max() == 3


# --- UnstructuredMesh -------------------------------------------------


_SQUARE_NODES = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
_SQUARE_ELEMENTS = np.array([[0, 1, 2], [0, 2, 3]])

_CORNER_TET4_NODES = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
_CORNER_TET4_ELEMENTS = np.array([[0, 1, 2, 3]])
_CORNER_TET_SURFACE_AREA = 3 * 0.5 + math.sqrt(3) / 2

_CORNER_TET10_NODES = np.array(
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
_CORNER_TET10_ELEMENTS = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])


def test_tri3_construction_and_geometry() -> None:
    m = UnstructuredMesh(_SQUARE_NODES, _SQUARE_ELEMENTS, "tri3")
    assert m.element_kind == "tri3"
    assert m.dim == 2
    assert m.n_nodes == 4
    assert m.n_elements == 2
    np.testing.assert_allclose(m.element_volumes.sum(), 1.0)
    f = m.boundary_faces()
    assert f.n_faces == 4
    assert f.area.sum() == pytest.approx(4.0)


def test_tet4_corner_construction_and_geometry() -> None:
    m = UnstructuredMesh(_CORNER_TET4_NODES, _CORNER_TET4_ELEMENTS, "tet4")
    assert m.element_kind == "tet4"
    assert m.dim == 3
    np.testing.assert_allclose(m.element_volumes, [1.0 / 6.0])
    np.testing.assert_allclose(m.element_centroids[0], _CORNER_TET4_NODES.mean(axis=0))
    f = m.boundary_faces()
    assert f.n_faces == 4
    assert f.nodes.shape == (4, 3)
    assert f.area.sum() == pytest.approx(_CORNER_TET_SURFACE_AREA)


def test_tet10_corner_construction_and_geometry() -> None:
    m = UnstructuredMesh(_CORNER_TET10_NODES, _CORNER_TET10_ELEMENTS, "tet10")
    assert m.element_kind == "tet10"
    assert m.dim == 3
    np.testing.assert_allclose(m.element_volumes, [1.0 / 6.0])
    np.testing.assert_allclose(m.element_centroids[0], _CORNER_TET10_NODES[:4].mean(axis=0))
    f = m.boundary_faces()
    assert f.n_faces == 4
    assert f.nodes.shape == (4, 6)
    assert f.area.sum() == pytest.approx(_CORNER_TET_SURFACE_AREA)


def test_unstructured_satisfies_mesh_protocol() -> None:
    m = UnstructuredMesh(_SQUARE_NODES, _SQUARE_ELEMENTS, "tri3")
    mesh: Mesh = m
    assert mesh.n_elements == 2


def test_unstructured_masks_default_all_design() -> None:
    m = UnstructuredMesh(_SQUARE_NODES, _SQUARE_ELEMENTS, "tri3")
    assert m.design.all()
    assert not m.solid.any()
    assert not m.void.any()


def test_unstructured_active_nodes_exclude_void_only_nodes() -> None:
    m = UnstructuredMesh(_SQUARE_NODES, _SQUARE_ELEMENTS, "tri3", void=[True, False])
    np.testing.assert_array_equal(m.active_elements, [False, True])
    np.testing.assert_array_equal(m.active_nodes, [True, False, True, True])
    np.testing.assert_array_equal(m.node_index_map, [0, -1, 1, 2])


def test_unstructured_is_immutable() -> None:
    m = UnstructuredMesh(_SQUARE_NODES, _SQUARE_ELEMENTS, "tri3")
    with pytest.raises(AttributeError):
        m.solid = np.ones(2, dtype=bool)  # type: ignore[misc]


def test_unstructured_derived_arrays_are_read_only() -> None:
    m = UnstructuredMesh(_SQUARE_NODES, _SQUARE_ELEMENTS, "tri3")
    for arr in (
        m.nodes,
        m.element_nodes,
        m.element_centroids,
        m.element_volumes,
        m.design,
        m.active_elements,
        m.boundary_faces().area,
        m.boundary_faces().nodes,
    ):
        with pytest.raises(ValueError, match="read-only"):
            arr[0] = 0


def test_unstructured_boundary_faces_cached() -> None:
    m = UnstructuredMesh(_SQUARE_NODES, _SQUARE_ELEMENTS, "tri3")
    assert m.boundary_faces() is m.boundary_faces()


def test_unstructured_repr_is_informative() -> None:
    m = UnstructuredMesh(_SQUARE_NODES, _SQUARE_ELEMENTS, "tri3", void=[True, False])
    r = repr(m)
    assert "tri3" in r
    assert "1 void" in r


def test_unstructured_negative_volume_raises_naming_element() -> None:
    elements = np.array([[0, 2, 1], [0, 2, 3]])  # element 0 wound clockwise
    with pytest.raises(MeshError, match=r"element ids \[0\]"):
        UnstructuredMesh(_SQUARE_NODES, elements, "tri3")


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_unstructured_non_finite_nodes_raise_naming_the_node(bad: float) -> None:
    # the signed-measure check cannot catch these on its own: `nan <= 0.0` is
    # False, and an inf coordinate can give a (meaningless) positive measure,
    # so without an explicit guard the mesh constructs and every derived
    # number -- volumes, centroids, stiffness, compliance -- is non-finite.
    nodes = _SQUARE_NODES.copy()
    nodes[2, 0] = bad
    with pytest.raises(MeshError, match=r"non-finite node coordinates for node ids \[2\]"):
        UnstructuredMesh(nodes, _SQUARE_ELEMENTS, "tri3")


def test_unstructured_out_of_range_connectivity_raises() -> None:
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    elements = np.array([[0, 1, 3]])
    with pytest.raises(MeshError, match="range"):
        UnstructuredMesh(nodes, elements, "tri3")


def test_unstructured_wrong_column_count_raises() -> None:
    elements = np.array([[0, 1, 2, 3]])
    with pytest.raises(MeshError, match="tri3"):
        UnstructuredMesh(_SQUARE_NODES, elements, "tri3")


def test_unstructured_unknown_kind_raises() -> None:
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    elements = np.array([[0, 1, 2]])
    with pytest.raises(MeshError, match="kind"):
        UnstructuredMesh(nodes, elements, "quad4")


def test_unstructured_dim_mismatch_raises() -> None:
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    elements = np.array([[0, 1, 2]])
    with pytest.raises(MeshError, match="dim"):
        UnstructuredMesh(nodes, elements, "tri3")


def test_unstructured_overlapping_masks_raise() -> None:
    both = np.array([True, False])
    with pytest.raises(MeshError, match="disjoint"):
        UnstructuredMesh(_SQUARE_NODES, _SQUARE_ELEMENTS, "tri3", solid=both, void=both)


def test_unstructured_wrong_mask_shape_raises() -> None:
    with pytest.raises(MeshError, match="shape"):
        UnstructuredMesh(_SQUARE_NODES, _SQUARE_ELEMENTS, "tri3", void=np.array([True]))


def test_unstructured_all_void_raises() -> None:
    with pytest.raises(MeshError, match="void"):
        UnstructuredMesh(_SQUARE_NODES, _SQUARE_ELEMENTS, "tri3", void=[True, True])


def test_unstructured_group_ids_out_of_range_raise() -> None:
    with pytest.raises(MeshError, match="group"):
        UnstructuredMesh(_SQUARE_NODES, _SQUARE_ELEMENTS, "tri3", groups={"bad": [5]})


def test_unstructured_groups_are_stored_and_read_only() -> None:
    m = UnstructuredMesh(_SQUARE_NODES, _SQUARE_ELEMENTS, "tri3", groups={"top": [1]})
    np.testing.assert_array_equal(m.groups["top"], [1])
    with pytest.raises(ValueError, match="read-only"):
        m.groups["top"][0] = 0


def test_unstructured_pickle_is_lean_and_preserves_immutability() -> None:
    import pickle

    m = UnstructuredMesh(
        _SQUARE_NODES,
        _SQUARE_ELEMENTS,
        "tri3",
        void=[True, False],
        groups={"top": [1]},
    )
    _ = m.element_centroids  # populate caches
    _ = m.boundary_faces()
    blob = pickle.dumps(m)
    m2 = pickle.loads(blob)

    assert "element_centroids" not in m2.__dict__  # caches are not serialized
    assert "design" not in m2.__dict__
    np.testing.assert_array_equal(m2.nodes, m.nodes)
    np.testing.assert_array_equal(m2.element_nodes, m.element_nodes)
    np.testing.assert_array_equal(m2.void, m.void)
    np.testing.assert_array_equal(m2.solid, m.solid)
    np.testing.assert_array_equal(m2.groups["top"], m.groups["top"])
    assert m2.boundary_faces().n_faces == m.boundary_faces().n_faces

    for arr in (m2.nodes, m2.element_nodes, m2.void, m2.solid, m2.groups["top"]):
        assert not arr.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            arr[0] = 0
