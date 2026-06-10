"""Tests for geometric selectors."""

import numpy as np
import pytest

from topokit.mesh import StructuredGrid
from topokit.selection import (
    Box,
    Cylinder,
    FaceSetSelector,
    NearPoint,
    OnBoundary,
    PlaneSlab,
    Predicate,
    SelectionError,
    Selector,
    Sphere,
    default_tolerance,
)

G22 = StructuredGrid(shape=(2, 2), spacing=(1.0, 1.0))


def test_default_tolerance_from_element_volumes() -> None:
    g = StructuredGrid(shape=(2, 2), spacing=(1.0, 0.25))
    # geometric-mean element size: (1.0 * 0.25) ** (1/2) = 0.5, halved
    assert default_tolerance(g) == pytest.approx(0.25)
    assert default_tolerance(G22) == pytest.approx(0.5)


def test_box_nodes_with_default_tol() -> None:
    np.testing.assert_array_equal(Box((0.0, 0.0), (0.0, 2.0)).nodes(G22), [0, 3, 6])


def test_box_elements() -> None:
    np.testing.assert_array_equal(Box((0.0, 0.0), (0.9, 2.0)).elements(G22), [0, 2])


def test_box_dim_mismatch_raises() -> None:
    with pytest.raises(SelectionError, match="dim"):
        Box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)).nodes(G22)


def test_sphere_single_node() -> None:
    np.testing.assert_array_equal(Sphere((1.0, 1.0), 0.1).nodes(G22), [4])


def test_plane_slab_explicit_tol() -> None:
    np.testing.assert_array_equal(
        PlaneSlab(point=(1.0, 0.0), normal=(1.0, 0.0), tol=0.1).nodes(G22), [1, 4, 7]
    )


def test_near_point_k1_and_deterministic_tie() -> None:
    np.testing.assert_array_equal(NearPoint((0.1, 0.1)).nodes(G22), [0])
    np.testing.assert_array_equal(NearPoint((0.1, 0.1), k=3).nodes(G22), [0, 1, 3])


def test_cylinder_3d() -> None:
    g = StructuredGrid(shape=(1, 1, 2), spacing=(1.0, 1.0, 1.0))
    cyl = Cylinder(p0=(0.5, 0.5, 0.0), p1=(0.5, 0.5, 2.0), radius=0.1)
    np.testing.assert_array_equal(cyl.elements(g), [0, 1])
    assert cyl.nodes(g).size == 0


def test_on_boundary_with_void_interface() -> None:
    g = StructuredGrid(shape=(2, 1), spacing=(1.0, 1.0), void=[False, True])
    sel = OnBoundary()
    np.testing.assert_array_equal(sel.faces(g), [0, 1, 2, 3])
    np.testing.assert_array_equal(sel.nodes(g), [0, 1, 3, 4])
    np.testing.assert_array_equal(sel.elements(g), [0])


def test_predicate() -> None:
    sel = Predicate(lambda c: c[:, 0] > 1.4)
    np.testing.assert_array_equal(sel.elements(G22), [1, 3])


def test_face_set_selector() -> None:
    g = StructuredGrid(shape=(1, 1), spacing=(1.0, 1.0))
    f = g.boundary_faces()
    minus_x = int(np.flatnonzero(f.normal[:, 0] < -0.5)[0])
    minus_y = int(np.flatnonzero(f.normal[:, 1] < -0.5)[0])
    sel = FaceSetSelector((minus_x, minus_y))
    np.testing.assert_array_equal(sel.faces(g), sorted([minus_x, minus_y]))
    np.testing.assert_array_equal(sel.nodes(g), [0, 1, 2])
    np.testing.assert_array_equal(sel.elements(g), [0])


def test_face_set_selector_out_of_range() -> None:
    g = StructuredGrid(shape=(1, 1), spacing=(1.0, 1.0))
    with pytest.raises(SelectionError, match="face id"):
        FaceSetSelector((9,)).faces(g)


def test_combinators_on_elements() -> None:
    left = Box((0.0, 0.0), (0.9, 2.0))
    bottom = Box((0.0, 0.0), (2.0, 0.9))
    np.testing.assert_array_equal((left | bottom).elements(G22), [0, 1, 2])
    np.testing.assert_array_equal((left & bottom).elements(G22), [0])
    np.testing.assert_array_equal((~left).elements(G22), [1, 3])


def test_complement_of_boundary_faces_is_empty() -> None:
    g = StructuredGrid(shape=(1, 1), spacing=(1.0, 1.0))
    assert (~OnBoundary()).faces(g).size == 0


def test_empty_selection_returns_empty_arrays() -> None:
    far = Sphere((100.0, 100.0), 0.1)
    assert far.nodes(G22).size == 0
    assert far.elements(G22).size == 0
    assert far.faces(G22).size == 0


def test_results_sorted_unique_int64() -> None:
    ids = (Box((0.0, 0.0), (2.0, 2.0)) | OnBoundary()).nodes(G22)
    assert ids.dtype == np.int64
    assert np.array_equal(ids, np.unique(ids))


def test_satisfies_protocol() -> None:
    sel: Selector = Box((0.0, 0.0), (1.0, 1.0))
    assert sel.nodes(G22).size > 0
