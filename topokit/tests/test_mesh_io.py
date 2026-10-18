# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Tests for `UnstructuredMesh.from_meshio` / `.from_file` (the `mesh` extra)."""

import sys
from pathlib import Path

import meshio
import numpy as np
import pytest

from topokit.mesh import MeshError, UnstructuredMesh

# Two positive-volume tets sharing a face, nodes 1,2,3.
_TWO_TET_NODES = np.array(
    [
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [0.0, 1.0, 0.0],  # 2
        [0.0, 0.0, 1.0],  # 3
        [1.0, 1.0, 1.0],  # 4
    ]
)
_TWO_TET_ELEMENTS = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])

_SQUARE_NODES_2D = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
_SQUARE_ELEMENTS = np.array([[0, 1, 2], [0, 2, 3]])

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

# Two separate, non-touching positive-volume tets, each its own cell block
# (mirrors gmsh 4.1: one CellBlock per geometrical entity).
_TWO_BLOCK_NODES = np.array(
    [
        [0.0, 0.0, 0.0],  # 0 tet A
        [1.0, 0.0, 0.0],  # 1
        [0.0, 1.0, 0.0],  # 2
        [0.0, 0.0, 1.0],  # 3
        [2.0, 0.0, 0.0],  # 4 tet B
        [3.0, 0.0, 0.0],  # 5
        [2.0, 1.0, 0.0],  # 6
        [2.0, 0.0, 1.0],  # 7
    ]
)
_TET_BLOCK_A = np.array([[0, 1, 2, 3]])
_TET_BLOCK_B = np.array([[4, 5, 6, 7]])


def test_from_meshio_tetra_with_cell_sets_resolves_groups_and_masks() -> None:
    mesh = meshio.Mesh(
        _TWO_TET_NODES,
        [("tetra", _TWO_TET_ELEMENTS)],
        cell_sets={"keep": [np.array([0])], "hole": [np.array([1])]},
    )
    m = UnstructuredMesh.from_meshio(mesh, solid="keep", void="hole")
    assert m.element_kind == "tet4"
    assert m.n_elements == 2
    np.testing.assert_array_equal(m.groups["keep"], [0])
    np.testing.assert_array_equal(m.groups["hole"], [1])
    np.testing.assert_array_equal(m.solid, [True, False])
    np.testing.assert_array_equal(m.void, [False, True])


def test_from_meshio_triangle_mesh_trims_zero_z_column() -> None:
    # meshio file formats store 3-column points even for planar meshes.
    points_3col = np.hstack([_SQUARE_NODES_2D, np.zeros((4, 1))])
    mesh = meshio.Mesh(points_3col, [("triangle", _SQUARE_ELEMENTS)])
    m = UnstructuredMesh.from_meshio(mesh)
    assert m.element_kind == "tri3"
    assert m.dim == 2
    assert m.nodes.shape == (4, 2)
    np.testing.assert_allclose(m.nodes, _SQUARE_NODES_2D)


def test_from_meshio_tetra10() -> None:
    mesh = meshio.Mesh(_CORNER_TET10_NODES, [("tetra10", _CORNER_TET10_ELEMENTS)])
    m = UnstructuredMesh.from_meshio(mesh)
    assert m.element_kind == "tet10"
    assert m.n_elements == 1
    np.testing.assert_allclose(m.element_volumes, [1.0 / 6.0])


def test_from_meshio_mixed_top_dimension_kinds_raises() -> None:
    # Degenerate but shape-valid hexahedron: the block-kind check runs before
    # any geometric validation, so this only needs 8 valid node ids.
    hexa = np.array([[0, 1, 4, 2, 0, 1, 4, 2]])
    mesh = meshio.Mesh(
        _TWO_TET_NODES,
        [("tetra", _TWO_TET_ELEMENTS), ("hexahedron", hexa)],
    )
    with pytest.raises(MeshError, match="hexahedron") as exc_info:
        UnstructuredMesh.from_meshio(mesh)
    assert "tetra" in str(exc_info.value)


def test_from_meshio_unknown_group_name_lists_available() -> None:
    mesh = meshio.Mesh(
        _TWO_TET_NODES,
        [("tetra", _TWO_TET_ELEMENTS)],
        cell_sets={"keep": [np.array([0])], "hole": [np.array([1])]},
    )
    with pytest.raises(MeshError, match="unknown solid group") as exc_info:
        UnstructuredMesh.from_meshio(mesh, solid="nope")
    msg = str(exc_info.value)
    assert "keep" in msg
    assert "hole" in msg


def test_from_meshio_two_same_kind_tetra_blocks_concatenates_and_resolves_cell_sets() -> None:
    # gmsh 4.1 emits one CellBlock per geometrical entity, so a standard
    # design-region + fixture model is two "tetra" blocks, not one. Cell
    # sets are keyed per block (dense empty arrays for blocks with no
    # members), mirroring what meshio's own readers (e.g. abaqus) produce.
    mesh = meshio.Mesh(
        _TWO_BLOCK_NODES,
        [("tetra", _TET_BLOCK_A), ("tetra", _TET_BLOCK_B)],
        cell_sets={
            "keep": [np.array([0]), np.array([], dtype=int)],
            "hole": [np.array([], dtype=int), np.array([0])],
        },
    )
    m = UnstructuredMesh.from_meshio(mesh, solid="keep", void="hole")
    assert m.element_kind == "tet4"
    assert m.n_elements == 2
    np.testing.assert_array_equal(m.element_nodes, np.vstack([_TET_BLOCK_A, _TET_BLOCK_B]))
    np.testing.assert_array_equal(m.groups["keep"], [0])
    np.testing.assert_array_equal(m.groups["hole"], [1])
    np.testing.assert_array_equal(m.solid, [True, False])
    np.testing.assert_array_equal(m.void, [False, True])


def test_from_meshio_two_same_kind_tetra_blocks_gmsh_tag_fallback() -> None:
    # Same two-block mesh, but named through gmsh physical tags instead of
    # cell_sets: cell_data["gmsh:physical"] is one array per block, so the
    # fallback must concatenate tags across both accepted blocks in order,
    # not just read the first block's tags.
    mesh = meshio.Mesh(
        _TWO_BLOCK_NODES,
        [("tetra", _TET_BLOCK_A), ("tetra", _TET_BLOCK_B)],
        cell_data={"gmsh:physical": [np.array([20]), np.array([30])]},
        field_data={"keep": np.array([20, 3]), "hole": np.array([30, 3])},
    )
    assert mesh.cell_sets_dict == {}
    m = UnstructuredMesh.from_meshio(mesh, solid="keep", void="hole")
    np.testing.assert_array_equal(m.groups["keep"], [0])
    np.testing.assert_array_equal(m.groups["hole"], [1])
    np.testing.assert_array_equal(m.solid, [True, False])
    np.testing.assert_array_equal(m.void, [False, True])


def test_from_meshio_gmsh_tag_resolution_is_dimension_scoped() -> None:
    # Physical Surface(1) "skin" and Physical Volume(1) "block" reuse tag 1
    # across dimensions; without dim-scoping, dict-insertion order decides
    # the winner and volume elements can resolve to the surface's name.
    mesh = meshio.Mesh(
        _TWO_TET_NODES,
        [("triangle", np.array([[0, 1, 2]])), ("tetra", _TWO_TET_ELEMENTS)],
        cell_data={"gmsh:physical": [np.array([1]), np.array([1, 1])]},
        field_data={"skin": np.array([1, 2]), "block": np.array([1, 3])},
    )
    m = UnstructuredMesh.from_meshio(mesh)
    assert set(m.groups) == {"block"}
    np.testing.assert_array_equal(sorted(m.groups["block"].tolist()), [0, 1])


def test_from_meshio_integer_void_array_with_non_mask_values_raises() -> None:
    # A 3-element mesh so ids can exceed 1; an id array whose length happens
    # to equal n_elements previously cast silently (and wrongly) to a
    # boolean mask instead of being rejected outright.
    elements = np.array([[0, 1, 2, 3], [1, 2, 3, 4], [0, 1, 2, 4]])
    mesh = meshio.Mesh(_TWO_TET_NODES, [("tetra", elements)])
    with pytest.raises(MeshError, match="element-id array"):
        UnstructuredMesh.from_meshio(mesh, void=np.array([0, 2, 1]))


def test_from_meshio_gmsh_physical_tags_fallback_when_no_cell_sets() -> None:
    # meshio's gmsh reader surfaces physical groups as a per-block
    # cell_data["gmsh:physical"] tag list plus field_data name -> [tag, dim],
    # not cell_sets_dict (verified empty for gmsh-format reads).
    cell_data = {"gmsh:physical": [np.array([10]), np.array([20, 30])]}
    field_data = {"skin": np.array([10, 2]), "keep": np.array([20, 3]), "hole": np.array([30, 3])}
    mesh = meshio.Mesh(
        _TWO_TET_NODES,
        [("triangle", np.array([[0, 1, 2]])), ("tetra", _TWO_TET_ELEMENTS)],
        cell_data=cell_data,
        field_data=field_data,
    )
    assert mesh.cell_sets_dict == {}  # confirms the fallback path is exercised
    m = UnstructuredMesh.from_meshio(mesh, solid="keep", void="hole")
    np.testing.assert_array_equal(m.groups["keep"], [0])
    np.testing.assert_array_equal(m.groups["hole"], [1])
    assert "skin" not in m.groups  # belongs to the ignored lower-dimension block
    np.testing.assert_array_equal(m.solid, [True, False])
    np.testing.assert_array_equal(m.void, [False, True])


def test_from_file_reads_vtk_tetra_mesh(tmp_path: Path) -> None:
    mesh = meshio.Mesh(_TWO_TET_NODES, [("tetra", _TWO_TET_ELEMENTS)])
    path = tmp_path / "mesh.vtk"
    meshio.write(path, mesh)

    m = UnstructuredMesh.from_file(path)
    assert m.element_kind == "tet4"
    assert m.n_elements == 2
    assert m.n_nodes == 5
    np.testing.assert_allclose(m.nodes, _TWO_TET_NODES)
    np.testing.assert_array_equal(m.element_nodes, _TWO_TET_ELEMENTS)


def test_from_file_missing_meshio_raises_with_extra_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "meshio", None)  # make `import meshio` raise
    with pytest.raises(ImportError, match=r"pip install topokit\[mesh\]"):
        UnstructuredMesh.from_file(tmp_path / "does-not-matter.msh")
