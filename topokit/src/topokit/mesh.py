# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Structured grids for the regular-mesh fast path.

Ordering conventions (binding for the whole codebase):

- Element id: ``i + nx*j (+ nx*ny*k)``, x-fastest.
- Node id: ``i + (nx+1)*j (+ (nx+1)*(ny+1)*k)``.
- quad4 connectivity (VTK_QUAD, CCW):
  ``[n(i,j), n(i+1,j), n(i+1,j+1), n(i,j+1)]``.
- hex8 connectivity (VTK_HEXAHEDRON): bottom face CCW, then the same four
  nodes at ``k+1``.

Element masks partition the grid into ``design`` (optimized), ``solid``
(fixed material) and ``void`` (removed from the system). Mesh topology is
host-side numpy; per-iteration arrays cross the array-backend boundary in
the physics modules, not here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

_F64 = npt.NDArray[np.float64]
_I64 = npt.NDArray[np.int64]
_Bool = npt.NDArray[np.bool_]


class MeshError(ValueError):
    """Invalid mesh construction or mask."""


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
