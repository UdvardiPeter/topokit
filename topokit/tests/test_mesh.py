"""Tests for the structured grid."""

import numpy as np
import pytest

from topokit.mesh import Mesh, MeshError, StructuredGrid


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
