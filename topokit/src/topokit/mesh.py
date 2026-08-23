# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Mesh discretizations: `StructuredGrid` and `UnstructuredMesh`.

`StructuredGrid` ordering conventions (binding for the whole codebase):

- Element id: ``i + nx*j (+ nx*ny*k)``, x-fastest.
- Node id: ``i + (nx+1)*j (+ (nx+1)*(ny+1)*k)``.
- quad4 connectivity (VTK_QUAD, CCW):
  ``[n(i,j), n(i+1,j), n(i+1,j+1), n(i,j+1)]``.
- hex8 connectivity (VTK_HEXAHEDRON): bottom face CCW, then the same four
  nodes at ``k+1``.

`UnstructuredMesh` connectivity follows VTK simplex ordering: tri3 nodes
CCW with positive signed area; tet4 nodes with positive signed volume
(``det(J) = det([x2-x1; x3-x1; x4-x1]) > 0``); tet10 is
VTK_QUADRATIC_TETRA (4 vertices, then midsides on edges
(0,1),(1,2),(2,0),(0,3),(1,3),(2,3)).

Element masks partition each mesh into ``design`` (optimized), ``solid``
(fixed material) and ``void`` (removed from the system). Mesh topology is
host-side numpy; per-iteration arrays cross the array-backend boundary in
the physics modules, not here.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import cached_property
from types import ModuleType
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
import numpy.typing as npt

_F64 = npt.NDArray[np.float64]
_I64 = npt.NDArray[np.int64]
_Bool = npt.NDArray[np.bool_]


class MeshError(ValueError):
    """Invalid mesh construction or mask."""


def _validate_mask(mask: npt.ArrayLike | None, n: int, name: str) -> _Bool:
    """Validate an element mask against element count ``n``, write-protected.

    Shared by `StructuredGrid` and `UnstructuredMesh` so both masks follow
    the same shape and dtype discipline.
    """
    if mask is None:
        out = np.zeros(n, dtype=bool)
    else:
        out = np.array(mask, dtype=bool, copy=True)
        if out.shape != (n,):
            raise MeshError(f"{name} mask shape {out.shape} != ({n},)")
    out.flags.writeable = False
    return out


@dataclass(frozen=True)
class BoundaryFaces:
    """Boundary faces of the active (non-void) domain, as flat arrays.

    In 2D, faces are edges with two nodes; in 3D, quads with four nodes.
    """

    owner: _I64
    normal: _F64
    area: _F64
    centroid: _F64
    nodes: _I64

    @property
    def n_faces(self) -> int:
        """Number of boundary faces."""
        return int(self.owner.size)


@runtime_checkable
class Mesh(Protocol):
    """Discretization interface the numerics are written against."""

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        ...

    @property
    def n_elements(self) -> int:
        """Number of elements."""
        ...

    @property
    def dim(self) -> int:
        """Spatial dimension, 2 or 3."""
        ...

    @property
    def element_kind(self) -> str:
        """Element identifier, e.g. ``"quad4"`` or ``"hex8"``.

        The fem module maps this to an element implementation; the mesh
        stays independent of L1.
        """
        ...

    @property
    def nodes(self) -> _F64:
        """Node coordinates, shape ``(n_nodes, dim)``."""
        ...

    @property
    def element_nodes(self) -> _I64:
        """Connectivity, shape ``(n_elements, nodes_per_element)``."""
        ...

    @property
    def element_volumes(self) -> _F64:
        """Element volumes (areas in 2D), shape ``(n_elements,)``."""
        ...

    @property
    def element_centroids(self) -> _F64:
        """Element centroids, shape ``(n_elements, dim)``."""
        ...

    @property
    def design(self) -> _Bool:
        """Elements whose density is optimized."""
        ...

    @property
    def solid(self) -> _Bool:
        """Elements with fixed material."""
        ...

    @property
    def void(self) -> _Bool:
        """Elements removed from the system."""
        ...

    @property
    def active_elements(self) -> _Bool:
        """Elements that are part of the system (non-void)."""
        ...

    @property
    def active_nodes(self) -> _Bool:
        """Nodes referenced by at least one non-void element."""
        ...

    @property
    def node_index_map(self) -> _I64:
        """Node id to condensed id; ``-1`` for inactive nodes."""
        ...

    def boundary_faces(self) -> BoundaryFaces:
        """Faces of non-void elements not shared with another non-void element.

        Must be deterministic and cheap on repeated calls; face ids
        (indices into the arrays) are stable identifiers.
        """
        ...


class StructuredGrid:
    """Regular quad/hex grid with anisotropic spacing and element masks.

    Instances are immutable after construction: geometry and masks are
    exposed as read-only properties, derived arrays are cached and
    write-protected. Masks are flat arrays in element-id (x-fastest) order.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        spacing: tuple[float, ...],
        origin: tuple[float, ...] | None = None,
        solid: npt.ArrayLike | None = None,
        void: npt.ArrayLike | None = None,
    ) -> None:
        self._shape = tuple(int(s) for s in shape)
        if len(self._shape) not in (2, 3):
            raise MeshError(f"dim must be 2 or 3, got {len(self._shape)}")
        if len(spacing) != len(self._shape):
            raise MeshError(
                f"spacing length ({len(spacing)}) must match shape length ({len(self._shape)})"
            )
        self._spacing = tuple(float(h) for h in spacing)
        if any(h <= 0.0 for h in self._spacing):
            raise MeshError(f"spacing must be positive, got {self._spacing}")
        if any(s < 1 for s in self._shape):
            raise MeshError(f"shape entries must be >= 1, got {self._shape}")
        if origin is None:
            origin = (0.0,) * len(self._shape)
        if len(origin) != len(self._shape):
            raise MeshError(
                f"origin length ({len(origin)}) must match shape length ({len(self._shape)})"
            )
        self._origin = tuple(float(x) for x in origin)

        self._solid = self._validated_mask(solid, "solid")
        self._void = self._validated_mask(void, "void")
        if bool((self._solid & self._void).any()):
            raise MeshError("solid and void masks must be disjoint")
        if bool(self._void.all()):
            raise MeshError("all elements are void")
        self._boundary_faces: BoundaryFaces | None = None

    def __repr__(self) -> str:
        roles = []
        if self._solid.any():
            roles.append(f"{int(self._solid.sum())} solid")
        if self._void.any():
            roles.append(f"{int(self._void.sum())} void")
        suffix = f", {', '.join(roles)}" if roles else ""
        return (
            f"StructuredGrid(shape={self.shape}, spacing={self.spacing}, "
            f"origin={self.origin}, {self.n_elements} elements{suffix})"
        )

    def __getstate__(self) -> dict[str, Any]:
        """Pickle only the defining state; caches rebuild on demand."""
        return {
            "_shape": self._shape,
            "_spacing": self._spacing,
            "_origin": self._origin,
            "_solid": self._solid,
            "_void": self._void,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        for mask in (self._solid, self._void):
            mask.flags.writeable = False  # numpy does not preserve the flag
        self._boundary_faces = None

    @property
    def shape(self) -> tuple[int, ...]:
        """Element counts per axis."""
        return self._shape

    @property
    def spacing(self) -> tuple[float, ...]:
        """Element size per axis."""
        return self._spacing

    @property
    def origin(self) -> tuple[float, ...]:
        """Coordinates of the first node."""
        return self._origin

    @property
    def solid(self) -> _Bool:
        """Elements with fixed material."""
        return self._solid

    @property
    def void(self) -> _Bool:
        """Elements removed from the system."""
        return self._void

    @classmethod
    def box(
        cls,
        size: tuple[float, ...],
        element_size: float | np.floating[Any] | np.integer[Any] | tuple[float, ...] | None = None,
        shape: tuple[int, ...] | None = None,
        origin: tuple[float, ...] | None = None,
    ) -> StructuredGrid:
        """Grid over a box of physical ``size``.

        Give either ``element_size`` (scalar or per-axis) or ``shape``.
        With ``element_size``, the element count per axis is rounded and the
        spacing recomputed so the grid covers ``size`` exactly.
        """
        if (element_size is None) == (shape is None):
            raise MeshError("give exactly one of element_size or shape")
        if shape is None:
            assert element_size is not None
            h = (
                (float(element_size),) * len(size)
                if isinstance(element_size, int | float | np.integer | np.floating)
                else tuple(float(x) for x in element_size)
            )
            if len(h) != len(size):
                raise MeshError(f"element_size length ({len(h)}) must match size ({len(size)})")
            shape = tuple(max(1, round(s / hi)) for s, hi in zip(size, h, strict=True))
        spacing = tuple(s / n for s, n in zip(size, shape, strict=True))
        return cls(shape=shape, spacing=spacing, origin=origin)

    def _validated_mask(self, mask: npt.ArrayLike | None, name: str) -> _Bool:
        if mask is None:
            out = np.zeros(self.n_elements, dtype=bool)
        else:
            out = np.array(mask, dtype=bool, copy=True)
            if out.shape != (self.n_elements,):
                raise MeshError(f"{name} mask shape {out.shape} != ({self.n_elements},)")
        out.flags.writeable = False
        return out

    @property
    def n_elements(self) -> int:
        """Number of elements."""
        return math.prod(self.shape)

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return math.prod(s + 1 for s in self.shape)

    @property
    def dim(self) -> int:
        """Spatial dimension."""
        return len(self.shape)

    @property
    def element_kind(self) -> str:
        """``"quad4"`` in 2D, ``"hex8"`` in 3D."""
        return "quad4" if self.dim == 2 else "hex8"

    @cached_property
    def design(self) -> _Bool:
        """Elements whose density is optimized."""
        out = ~(self._solid | self._void)
        out.flags.writeable = False
        return out

    @cached_property
    def active_elements(self) -> _Bool:
        """Elements that are part of the system (non-void)."""
        out = ~self._void
        out.flags.writeable = False
        return out

    @cached_property
    def nodes(self) -> _F64:
        """Node coordinates, shape ``(n_nodes, dim)``, x-fastest ordering."""
        axes = [
            self.origin[a] + self.spacing[a] * np.arange(self.shape[a] + 1) for a in range(self.dim)
        ]
        grids = np.meshgrid(*axes, indexing="ij")
        out = np.stack([g.ravel(order="F") for g in grids], axis=1)
        out.flags.writeable = False
        return out

    def _element_index_grids(self) -> list[_I64]:
        idx = [np.arange(s, dtype=np.int64) for s in self.shape]
        return [g.ravel(order="F") for g in np.meshgrid(*idx, indexing="ij")]

    def _corner_node_ids(self, index_arrays: list[_I64]) -> _I64:
        nx1 = self.shape[0] + 1
        out = index_arrays[0] + nx1 * index_arrays[1]
        if self.dim == 3:
            out = out + nx1 * (self.shape[1] + 1) * index_arrays[2]
        return out

    @cached_property
    def _node_offsets(self) -> _I64:
        x = self.shape[0] + 1
        if self.dim == 2:
            return np.array([0, 1, 1 + x, x], dtype=np.int64)
        layer = x * (self.shape[1] + 1)
        bottom = [0, 1, 1 + x, x]
        return np.array(bottom + [o + layer for o in bottom], dtype=np.int64)

    @cached_property
    def element_nodes(self) -> _I64:
        """Connectivity in VTK ordering, shape ``(n_elements, 4 or 8)``."""
        corner = self._corner_node_ids(self._element_index_grids())
        out = corner[:, None] + self._node_offsets[None, :]
        out.flags.writeable = False
        return out

    @cached_property
    def element_centroids(self) -> _F64:
        """Element centroids, shape ``(n_elements, dim)``."""
        grids = self._element_index_grids()
        cols = [self.origin[a] + (grids[a] + 0.5) * self.spacing[a] for a in range(self.dim)]
        out = np.stack(cols, axis=1)
        out.flags.writeable = False
        return out

    @cached_property
    def element_volumes(self) -> _F64:
        """Constant element volume, shape ``(n_elements,)``."""
        out = np.full(self.n_elements, math.prod(self.spacing))
        out.flags.writeable = False
        return out

    @cached_property
    def active_nodes(self) -> _Bool:
        """Nodes referenced by at least one non-void element."""
        active = np.zeros(self.n_nodes, dtype=bool)
        active[self.element_nodes[self.active_elements].ravel()] = True
        active.flags.writeable = False
        return active

    @cached_property
    def node_index_map(self) -> _I64:
        """Old node id to condensed id; ``-1`` for inactive nodes."""
        out = np.full(self.n_nodes, -1, dtype=np.int64)
        out[self.active_nodes] = np.arange(int(self.active_nodes.sum()), dtype=np.int64)
        out.flags.writeable = False
        return out

    def to_grid(self, flat: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """View a flat element array as the ``(nx, ny[, nz])`` grid.

        Centralizes the x-fastest (Fortran-order) convention. Never reshape
        element arrays manually; a forgotten ``order="F"`` scrambles fields
        silently.
        """
        arr = np.asarray(flat)
        if arr.shape != (self.n_elements,):
            raise MeshError(f"expected flat shape ({self.n_elements},), got {arr.shape}")
        return arr.reshape(self.shape, order="F")

    def to_flat(self, grid: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Flatten a ``(nx, ny[, nz])`` grid array back to element-id order."""
        arr = np.asarray(grid)
        if arr.shape != self.shape:
            raise MeshError(f"expected grid shape {self.shape}, got {arr.shape}")
        return arr.ravel(order="F")

    def _face_node_offsets(self, axis: int, positive: bool) -> _I64:
        x = self.shape[0] + 1
        if self.dim == 2:
            offsets = {
                0: np.array([0, x], dtype=np.int64),
                1: np.array([0, 1], dtype=np.int64),
            }[axis]
            shift = {0: 1, 1: x}[axis]
        else:
            layer = x * (self.shape[1] + 1)
            offsets = {
                0: np.array([0, x, x + layer, layer], dtype=np.int64),
                1: np.array([0, 1, 1 + layer, layer], dtype=np.int64),
                2: np.array([0, 1, 1 + x, x], dtype=np.int64),
            }[axis]
            shift = {0: 1, 1: x, 2: layer}[axis]
        return offsets + (shift if positive else 0)

    def boundary_faces(self) -> BoundaryFaces:
        """Faces of non-void elements not shared with another non-void element.

        Computed once and cached; face ids (indices into the arrays) are
        stable. Face nodes are in perimeter order; winding is unspecified,
        the outward normal is explicit.
        """
        if self._boundary_faces is None:
            self._boundary_faces = self._compute_boundary_faces()
        return self._boundary_faces

    def _compute_boundary_faces(self) -> BoundaryFaces:
        active = self.to_grid(self.active_elements)
        volume = math.prod(self.spacing)
        owners: list[_I64] = []
        normals: list[_F64] = []
        areas: list[_F64] = []
        centroids: list[_F64] = []
        nodes: list[_I64] = []
        for axis in range(self.dim):
            for sign in (-1, 1):
                neighbor = np.zeros_like(active)
                dst = [slice(None)] * self.dim
                src = [slice(None)] * self.dim
                if sign < 0:
                    dst[axis], src[axis] = slice(1, None), slice(None, -1)
                else:
                    dst[axis], src[axis] = slice(None, -1), slice(1, None)
                neighbor[tuple(dst)] = active[tuple(src)]
                where = np.nonzero(active & ~neighbor)
                index_arrays = [w.astype(np.int64) for w in where]
                owner = index_arrays[0] + self.shape[0] * index_arrays[1]
                if self.dim == 3:
                    owner = owner + self.shape[0] * self.shape[1] * index_arrays[2]
                corner = self._corner_node_ids(index_arrays)
                face_nodes = corner[:, None] + self._face_node_offsets(axis, sign > 0)[None, :]
                normal = np.zeros((owner.size, self.dim))
                normal[:, axis] = float(sign)
                centroid = np.stack(
                    [
                        self.origin[a] + (index_arrays[a] + 0.5) * self.spacing[a]
                        for a in range(self.dim)
                    ],
                    axis=1,
                )
                centroid[:, axis] += sign * self.spacing[axis] / 2.0
                owners.append(owner)
                normals.append(normal)
                areas.append(np.full(owner.size, volume / self.spacing[axis]))
                centroids.append(centroid)
                nodes.append(face_nodes)
        owner = np.concatenate(owners)
        normal = np.concatenate(normals)
        area = np.concatenate(areas)
        centroid = np.concatenate(centroids)
        face_nodes_all = np.concatenate(nodes)
        for arr in (owner, normal, area, centroid, face_nodes_all):
            arr.flags.writeable = False
        return BoundaryFaces(
            owner=owner, normal=normal, area=area, centroid=centroid, nodes=face_nodes_all
        )


# element_kind -> connectivity column count
_ELEMENT_KIND_COLUMNS: dict[str, int] = {"tri3": 3, "tet4": 4, "tet10": 10}
# element_kind -> required node dimension
_ELEMENT_KIND_DIM: dict[str, int] = {"tri3": 2, "tet4": 3, "tet10": 3}
# element_kind -> corner (vertex) count; tet10's remaining columns are midsides
_ELEMENT_KIND_CORNERS: dict[str, int] = {"tri3": 3, "tet4": 4, "tet10": 4}

# element_kind -> corner-local-index patterns of its boundary facets (edges
# in 2D, triangular faces in 3D). Initial winding is arbitrary; boundary_faces
# reorients each retained facet outward, so a fixed enumeration order suffices.
_FACE_CORNER_PATTERNS: dict[str, tuple[tuple[int, ...], ...]] = {
    "tri3": ((0, 1), (1, 2), (2, 0)),
    "tet4": ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)),
    "tet10": ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)),
}

# VTK_QUADRATIC_TETRA midside column, keyed by the unordered pair of
# corner-local indices whose edge midpoint it sits on.
_TET10_MIDSIDE_COLUMN: dict[frozenset[int], int] = {
    frozenset((0, 1)): 4,
    frozenset((1, 2)): 5,
    frozenset((2, 0)): 6,
    frozenset((0, 3)): 7,
    frozenset((1, 3)): 8,
    frozenset((2, 3)): 9,
}


def _signed_measure(nodes: _F64, elements: _I64, kind: str, dim: int) -> _F64:
    """Signed area (2D) or signed volume (3D) of each element's corners.

    ``tet10`` uses the corner tetrahedron; exact for straight edges.
    """
    n_corners = _ELEMENT_KIND_CORNERS[kind]
    p = nodes[elements[:, :n_corners]]
    e1 = p[:, 1] - p[:, 0]
    e2 = p[:, 2] - p[:, 0]
    if dim == 2:
        out = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    else:
        e3 = p[:, 3] - p[:, 0]
        out = (
            e1[:, 0] * (e2[:, 1] * e3[:, 2] - e2[:, 2] * e3[:, 1])
            - e1[:, 1] * (e2[:, 0] * e3[:, 2] - e2[:, 2] * e3[:, 0])
            + e1[:, 2] * (e2[:, 0] * e3[:, 1] - e2[:, 1] * e3[:, 0])
        ) / 6.0
    return np.asarray(out, dtype=np.float64)


# meshio cell-type name -> topokit element kind
_MESHIO_CELL_KIND: dict[str, str] = {"triangle": "tri3", "tetra": "tet4", "tetra10": "tet10"}

MESH_HINT = "reading external mesh files needs the [mesh] extra: pip install topokit[mesh]"


def _require_meshio() -> ModuleType:
    """Import meshio or raise with the ``[mesh]`` install hint."""
    try:
        import meshio
    except ImportError as exc:
        raise ImportError(MESH_HINT) from exc
    return cast(ModuleType, meshio)


def _accepted_meshio_block(mesh: Any) -> tuple[int, Any]:
    """Find the mesh's sole top-dimension cell block, if it is a supported kind.

    Lower-dimension blocks (e.g. boundary triangles alongside a tet volume)
    are ignored. More than one block at the top dimension, or a top-dimension
    block whose type is not in `_MESHIO_CELL_KIND`, raises.
    """
    top_dim = max((block.dim for block in mesh.cells), default=-1)
    candidates = [(i, block) for i, block in enumerate(mesh.cells) if block.dim == top_dim]
    if len(candidates) != 1 or candidates[0][1].type not in _MESHIO_CELL_KIND:
        kinds = sorted({block.type for _i, block in candidates})
        raise MeshError(
            f"expected exactly one supported cell block ({sorted(_MESHIO_CELL_KIND)}) "
            f"at the mesh's top dimension ({top_dim}D); found: {kinds}"
        )
    return candidates[0]


def _meshio_groups(mesh: Any, block_index: int, cell_type: str) -> dict[str, npt.ArrayLike]:
    """Named element-id groups for ``cell_type``, from cell sets or gmsh tags.

    Prefers `mesh.cell_sets_dict` entries restricted to ``cell_type``
    (element ids already refer to positions within the single accepted
    block). Falls back to gmsh's ``cell_data["gmsh:physical"]`` tags for the
    accepted block, resolved to names through `mesh.field_data`
    (``{name: [tag, dim]}``); tags with no matching name fall back to their
    numeric string.
    """
    groups: dict[str, npt.ArrayLike] = {}
    for name, by_type in mesh.cell_sets_dict.items():
        ids = by_type.get(cell_type)
        if ids is not None and ids.size:
            groups[name] = np.asarray(ids, dtype=np.int64)
    if groups:
        return groups

    tags_by_block = mesh.cell_data.get("gmsh:physical")
    if not tags_by_block:
        return groups
    tags = np.asarray(tags_by_block[block_index])
    tag_to_name = {int(tag): name for name, (tag, _dim) in mesh.field_data.items()}
    for tag in np.unique(tags).tolist():
        name = tag_to_name.get(int(tag), str(int(tag)))
        groups[name] = np.nonzero(tags == tag)[0].astype(np.int64)
    return groups


def _meshio_nodes(points: npt.ArrayLike, kind: str) -> _F64:
    """Node coordinates trimmed to ``kind``'s dimension.

    File formats (gmsh, vtk, ...) store points with 3 columns even for
    planar meshes, with an all-zero 3rd column; drop it so a ``tri3`` mesh
    matches the constructor's 2D requirement. Genuinely 3D data (nonzero
    extra columns) is passed through unchanged and left to the
    constructor's own dimension check to reject.
    """
    arr = np.asarray(points, dtype=np.float64)
    expected_dim = _ELEMENT_KIND_DIM[kind]
    if arr.ndim == 2 and arr.shape[1] > expected_dim and np.allclose(arr[:, expected_dim:], 0.0):
        return np.ascontiguousarray(arr[:, :expected_dim])
    return arr


class UnstructuredMesh:
    """Simplex mesh (tri3, tet4, or tet10) with element masks.

    Instances are immutable after construction: geometry and masks are
    exposed as read-only properties, derived arrays are cached and
    write-protected, mirroring `StructuredGrid`. Connectivity follows VTK
    node ordering; ``tet10`` is VTK_QUADRATIC_TETRA (4 vertices, then
    midsides on edges (0,1),(1,2),(2,0),(0,3),(1,3),(2,3)). Elements must
    have positive signed area/volume; construction raises rather than
    silently reorienting.
    """

    def __init__(
        self,
        nodes: npt.ArrayLike,
        elements: npt.ArrayLike,
        element_kind: str,
        *,
        solid: npt.ArrayLike | None = None,
        void: npt.ArrayLike | None = None,
        groups: dict[str, npt.ArrayLike] | None = None,
    ) -> None:
        if element_kind not in _ELEMENT_KIND_COLUMNS:
            raise MeshError(
                f"unknown element kind {element_kind!r}; supported: {sorted(_ELEMENT_KIND_COLUMNS)}"
            )
        self._kind = element_kind

        nodes_arr = np.array(nodes, dtype=np.float64, copy=True)
        if nodes_arr.ndim != 2:
            raise MeshError(f"nodes must have shape (n_nodes, dim), got {nodes_arr.shape}")
        dim = int(nodes_arr.shape[1])
        expected_dim = _ELEMENT_KIND_DIM[element_kind]
        if dim != expected_dim:
            raise MeshError(
                f"element kind {element_kind!r} requires {expected_dim}D nodes, got dim {dim}"
            )
        self._dim = dim

        elements_arr = np.array(elements, dtype=np.int64, copy=True)
        expected_cols = _ELEMENT_KIND_COLUMNS[element_kind]
        if elements_arr.ndim != 2 or elements_arr.shape[1] != expected_cols:
            raise MeshError(
                f"{element_kind!r} connectivity must have shape "
                f"(n_elements, {expected_cols}), got {elements_arr.shape}"
            )
        n_nodes = nodes_arr.shape[0]
        if elements_arr.size and (
            int(elements_arr.min()) < 0 or int(elements_arr.max()) >= n_nodes
        ):
            raise MeshError(
                f"connectivity references node ids outside [0, {n_nodes}); "
                f"got range [{int(elements_arr.min())}, {int(elements_arr.max())}]"
            )

        signed = _signed_measure(nodes_arr, elements_arr, element_kind, dim)
        bad = np.nonzero(signed <= 0.0)[0]
        if bad.size:
            measure = "area" if dim == 2 else "volume"
            raise MeshError(f"non-positive signed {measure} for element ids {bad.tolist()}")

        nodes_arr.flags.writeable = False
        elements_arr.flags.writeable = False
        self._nodes = nodes_arr
        self._elements = elements_arr

        self._solid = _validate_mask(solid, self.n_elements, "solid")
        self._void = _validate_mask(void, self.n_elements, "void")
        if bool((self._solid & self._void).any()):
            raise MeshError("solid and void masks must be disjoint")
        if bool(self._void.all()):
            raise MeshError("all elements are void")

        self._groups = self._validated_groups(groups)
        self._boundary_faces: BoundaryFaces | None = None

    def __repr__(self) -> str:
        roles = []
        if self._solid.any():
            roles.append(f"{int(self._solid.sum())} solid")
        if self._void.any():
            roles.append(f"{int(self._void.sum())} void")
        suffix = f", {', '.join(roles)}" if roles else ""
        return (
            f"UnstructuredMesh(kind={self._kind!r}, {self.n_elements} elements, "
            f"{self.n_nodes} nodes{suffix})"
        )

    def __getstate__(self) -> dict[str, Any]:
        """Pickle only the defining state; caches rebuild on demand."""
        return {
            "_kind": self._kind,
            "_dim": self._dim,
            "_nodes": self._nodes,
            "_elements": self._elements,
            "_solid": self._solid,
            "_void": self._void,
            "_groups": self._groups,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        for arr in (self._nodes, self._elements, self._solid, self._void, *self._groups.values()):
            arr.flags.writeable = False  # numpy does not preserve the flag
        self._boundary_faces = None

    def _validated_groups(self, groups: dict[str, npt.ArrayLike] | None) -> dict[str, _I64]:
        if groups is None:
            return {}
        out: dict[str, _I64] = {}
        for name, ids in groups.items():
            arr = np.array(ids, dtype=np.int64, copy=True)
            if arr.size and (int(arr.min()) < 0 or int(arr.max()) >= self.n_elements):
                raise MeshError(
                    f"group {name!r} references element ids outside [0, {self.n_elements})"
                )
            arr.flags.writeable = False
            out[name] = arr
        return out

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return int(self._nodes.shape[0])

    @property
    def n_elements(self) -> int:
        """Number of elements."""
        return int(self._elements.shape[0])

    @property
    def dim(self) -> int:
        """Spatial dimension."""
        return self._dim

    @property
    def element_kind(self) -> str:
        """``"tri3"``, ``"tet4"``, or ``"tet10"``, as given to the constructor."""
        return self._kind

    @property
    def nodes(self) -> _F64:
        """Node coordinates, shape ``(n_nodes, dim)``."""
        return self._nodes

    @property
    def element_nodes(self) -> _I64:
        """Connectivity, shape ``(n_elements, nodes_per_element)``."""
        return self._elements

    @property
    def solid(self) -> _Bool:
        """Elements with fixed material."""
        return self._solid

    @property
    def void(self) -> _Bool:
        """Elements removed from the system."""
        return self._void

    @property
    def groups(self) -> dict[str, _I64]:
        """Named element-id sets, e.g. physical groups from an imported mesh."""
        return dict(self._groups)

    @classmethod
    def from_meshio(
        cls,
        mesh: Any,
        *,
        solid: npt.ArrayLike | str | None = None,
        void: npt.ArrayLike | str | None = None,
    ) -> UnstructuredMesh:
        """Build an `UnstructuredMesh` from an in-memory ``meshio.Mesh``.

        Exactly one cell block at the mesh's top topological dimension must
        be a supported kind (``"triangle"`` -> tri3, ``"tetra"`` -> tet4,
        ``"tetra10"`` -> tet10); a mesh whose top-dimension blocks are mixed
        or unsupported raises `MeshError` listing the kinds found.
        Lower-dimension blocks (e.g. boundary triangles alongside a tet
        volume) are ignored.

        Named element groups are resolved from ``mesh.cell_sets_dict``
        restricted to the accepted block when present, else from gmsh
        physical tags, and become the constructor's ``groups=``.
        ``solid``/``void`` accept either a group name, resolved against
        those groups (an unknown name raises `MeshError` listing the
        available ones), or an explicit element mask/id array passed
        through unchanged.
        """
        block_index, block = _accepted_meshio_block(mesh)
        kind = _MESHIO_CELL_KIND[block.type]
        nodes = _meshio_nodes(mesh.points, kind)
        groups = _meshio_groups(mesh, block_index, block.type)

        def _resolve(value: npt.ArrayLike | str | None) -> npt.ArrayLike | None:
            if not isinstance(value, str):
                return value
            if value not in groups:
                raise MeshError(f"unknown group {value!r}; available: {sorted(groups)}")
            mask = np.zeros(block.data.shape[0], dtype=bool)
            mask[np.asarray(groups[value], dtype=np.int64)] = True
            return mask

        return cls(
            nodes,
            block.data,
            kind,
            solid=_resolve(solid),
            void=_resolve(void),
            groups=groups,
        )

    @classmethod
    def from_file(
        cls,
        path: str | os.PathLike[str],
        *,
        solid: npt.ArrayLike | str | None = None,
        void: npt.ArrayLike | str | None = None,
    ) -> UnstructuredMesh:
        """Read a mesh file through meshio and build an `UnstructuredMesh`.

        Equivalent to ``UnstructuredMesh.from_meshio(meshio.read(path), ...)``.
        Raises `ImportError` naming the ``[mesh]`` extra if meshio is not
        installed.
        """
        meshio = _require_meshio()
        return cls.from_meshio(meshio.read(path), solid=solid, void=void)

    @cached_property
    def design(self) -> _Bool:
        """Elements whose density is optimized."""
        out = ~(self._solid | self._void)
        out.flags.writeable = False
        return out

    @cached_property
    def active_elements(self) -> _Bool:
        """Elements that are part of the system (non-void)."""
        out = ~self._void
        out.flags.writeable = False
        return out

    @cached_property
    def element_volumes(self) -> _F64:
        """Element volumes (areas in 2D), shape ``(n_elements,)``.

        ``tet10`` volumes use the corner tetrahedron; exact for straight
        edges.
        """
        out = _signed_measure(self._nodes, self._elements, self._kind, self._dim)
        out.flags.writeable = False
        return out

    @cached_property
    def element_centroids(self) -> _F64:
        """Element centroids, shape ``(n_elements, dim)``.

        ``tet10`` centroids use the corner tetrahedron; exact for straight
        edges.
        """
        n_corners = _ELEMENT_KIND_CORNERS[self._kind]
        out = np.asarray(self._nodes[self._elements[:, :n_corners]].mean(axis=1), dtype=np.float64)
        out.flags.writeable = False
        return out

    @cached_property
    def active_nodes(self) -> _Bool:
        """Nodes referenced by at least one non-void element."""
        active = np.zeros(self.n_nodes, dtype=bool)
        active[self._elements[self.active_elements].ravel()] = True
        active.flags.writeable = False
        return active

    @cached_property
    def node_index_map(self) -> _I64:
        """Old node id to condensed id; ``-1`` for inactive nodes."""
        out = np.full(self.n_nodes, -1, dtype=np.int64)
        out[self.active_nodes] = np.arange(int(self.active_nodes.sum()), dtype=np.int64)
        out.flags.writeable = False
        return out

    def boundary_faces(self) -> BoundaryFaces:
        """Faces of non-void elements not shared with another non-void element.

        Computed once and cached; face ids (indices into the arrays) are
        stable, ordered by each face's sorted corner-node-id tuple. Corner
        nodes are ordered so the normal points away from the owner; tet10
        faces additionally carry the 3 midside nodes of their corner edges,
        appended after the 3 corners.
        """
        if self._boundary_faces is None:
            self._boundary_faces = self._compute_boundary_faces()
        return self._boundary_faces

    def _compute_boundary_faces(self) -> BoundaryFaces:
        patterns = _FACE_CORNER_PATTERNS[self._kind]
        active_ids = np.nonzero(self.active_elements)[0]
        occurrences: dict[tuple[int, ...], list[tuple[int, tuple[int, ...]]]] = {}
        for e in active_ids.tolist():
            conn = self._elements[e]
            for pattern in patterns:
                corners = tuple(int(conn[i]) for i in pattern)
                key = tuple(sorted(corners))
                occurrences.setdefault(key, []).append((e, pattern))

        boundary = sorted(
            ((key, occ[0][0], occ[0][1]) for key, occ in occurrences.items() if len(occ) == 1),
            key=lambda item: item[0],
        )

        n_corner = len(patterns[0])
        n_face_nodes = n_corner + (3 if self._kind == "tet10" else 0)
        n_faces = len(boundary)
        owner = np.empty(n_faces, dtype=np.int64)
        normal = np.empty((n_faces, self._dim), dtype=np.float64)
        area = np.empty(n_faces, dtype=np.float64)
        centroid = np.empty((n_faces, self._dim), dtype=np.float64)
        face_nodes = np.empty((n_faces, n_face_nodes), dtype=np.int64)

        for row, (_key, e, pattern) in enumerate(boundary):
            conn = self._elements[e]
            local = list(pattern)
            global_ids = [int(conn[i]) for i in local]
            coords = self._nodes[global_ids]
            face_centroid = coords.mean(axis=0)
            unit, face_area = self._face_normal_area(coords)
            if np.dot(unit, face_centroid - self.element_centroids[e]) < 0.0:
                local.reverse()
                global_ids.reverse()
                coords = self._nodes[global_ids]
                face_centroid = coords.mean(axis=0)
                unit, face_area = self._face_normal_area(coords)

            owner[row] = e
            normal[row] = unit
            area[row] = face_area
            centroid[row] = face_centroid
            if self._kind == "tet10":
                pairs = zip(local, local[1:] + local[:1], strict=True)
                midsides = [int(conn[_TET10_MIDSIDE_COLUMN[frozenset(p)]]) for p in pairs]
                face_nodes[row] = global_ids + midsides
            else:
                face_nodes[row] = global_ids

        for arr in (owner, normal, area, centroid, face_nodes):
            arr.flags.writeable = False
        return BoundaryFaces(
            owner=owner, normal=normal, area=area, centroid=centroid, nodes=face_nodes
        )

    @staticmethod
    def _face_normal_area(coords: _F64) -> tuple[_F64, float]:
        """Unnormalized-then-unit normal and area of a straight facet.

        2D: an edge (2 points); the raw normal is a 90-degree rotation of
        the edge vector, so its length equals the edge length. 3D: a
        triangle (3 points); the raw normal is the corner cross product,
        with area half its length.
        """
        if coords.shape[0] == 2:
            edge = coords[1] - coords[0]
            raw = np.array([edge[1], -edge[0]], dtype=np.float64)
            face_area = float(np.linalg.norm(edge))
        else:
            raw = np.cross(coords[1] - coords[0], coords[2] - coords[0])
            face_area = 0.5 * float(np.linalg.norm(raw))
        unit = raw / float(np.linalg.norm(raw))
        return unit, face_area
