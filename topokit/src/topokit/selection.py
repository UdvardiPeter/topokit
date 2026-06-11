# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Geometric selectors, the single mechanism for "where".

Loads, supports, regions, and CAD face picks all resolve through selectors.
A selector answers ``nodes(mesh)``, ``elements(mesh)`` and ``faces(mesh)``
as sorted unique int64 id arrays; face ids index into
``mesh.boundary_faces()``.

Inclusion is geometric: an entity is selected when its representative point
(node coordinate, element centroid, face centroid) lies inside the shape
expanded by ``tol``. The default tolerance is half a typical element size,
see :func:`default_tolerance`. Selection does not intersect with active
masks and an empty result is not an error; both are the consumer's job.

Combine selectors with ``&``, ``|`` and ``~``. ``~`` complements within the
mesh's full id range per entity type. To write a custom selector, subclass
:class:`SelectorBase` and implement ``_mask``; the :class:`Selector`
protocol is the consumption contract.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from topokit.mesh import Mesh

_F64 = npt.NDArray[np.float64]
_I64 = npt.NDArray[np.int64]
_Bool = npt.NDArray[np.bool_]


class SelectionError(ValueError):
    """Invalid selector parameters or use."""


def default_tolerance(mesh: Mesh) -> float:
    """Half a typical element size: ``0.5 * median(element_volumes) ** (1/dim)``.

    The scale is isotropic. On strongly anisotropic grids it over-selects
    along the fine axis; pass an explicit ``tol`` sized for the relevant
    axis there.
    """
    return float(0.5 * float(np.median(mesh.element_volumes)) ** (1.0 / mesh.dim))


@runtime_checkable
class Selector(Protocol):
    """Resolves entity ids on a mesh."""

    def nodes(self, mesh: Mesh) -> _I64:
        """Sorted unique node ids."""
        ...

    def elements(self, mesh: Mesh) -> _I64:
        """Sorted unique element ids."""
        ...

    def faces(self, mesh: Mesh) -> _I64:
        """Sorted unique boundary-face ids."""
        ...


class SelectorBase:
    """Base for selectors defined by a point-set membership test.

    Implement ``_mask``; ``nodes``/``elements``/``faces`` apply it to the
    matching representative points. Provides ``&``, ``|``, ``~``.
    """

    def nodes(self, mesh: Mesh) -> _I64:
        """Sorted unique node ids."""
        return self._ids(mesh.nodes, mesh)

    def elements(self, mesh: Mesh) -> _I64:
        """Sorted unique element ids."""
        return self._ids(mesh.element_centroids, mesh)

    def faces(self, mesh: Mesh) -> _I64:
        """Sorted unique boundary-face ids."""
        return self._ids(mesh.boundary_faces().centroid, mesh)

    def element_mask(self, mesh: Mesh) -> _Bool:
        """Boolean element mask, the bridge to mesh ``solid``/``void`` regions."""
        out = np.zeros(mesh.n_elements, dtype=bool)
        out[self.elements(mesh)] = True
        return out

    def node_mask(self, mesh: Mesh) -> _Bool:
        """Boolean node mask of the selection."""
        out = np.zeros(mesh.n_nodes, dtype=bool)
        out[self.nodes(mesh)] = True
        return out

    def _ids(self, coords: _F64, mesh: Mesh) -> _I64:
        return np.flatnonzero(self._mask(coords, mesh)).astype(np.int64)

    def _mask(self, coords: _F64, mesh: Mesh) -> _Bool:
        raise NotImplementedError

    def __and__(self, other: Selector) -> SelectorBase:
        return _And(self, other)

    def __or__(self, other: Selector) -> SelectorBase:
        return _Or(self, other)

    def __invert__(self) -> SelectorBase:
        return _Not(self)


def _check_dim(point: tuple[float, ...], coords: _F64) -> None:
    if len(point) != coords.shape[1]:
        raise SelectionError(f"selector dim {len(point)} != mesh dim {coords.shape[1]}")


def _freeze(obj: object, **fields: object) -> None:
    """Coerce constructor arguments on a frozen dataclass.

    Lists and numpy scalars are common call-site inputs; coercing to plain
    tuples and floats keeps selectors hashable and equality well-behaved.
    """
    for name, value in fields.items():
        object.__setattr__(obj, name, value)


def _tol(tol: float | None, mesh: Mesh) -> float:
    return default_tolerance(mesh) if tol is None else tol


@dataclass(frozen=True)
class Box(SelectorBase):
    """Axis-aligned box from ``lower`` to ``upper`` corner."""

    lower: tuple[float, ...]
    upper: tuple[float, ...]
    tol: float | None = None

    def __post_init__(self) -> None:
        _freeze(
            self,
            lower=tuple(float(x) for x in self.lower),
            upper=tuple(float(x) for x in self.upper),
            tol=None if self.tol is None else float(self.tol),
        )
        if len(self.lower) != len(self.upper):
            raise SelectionError("lower and upper must have the same length")
        if any(lo > hi for lo, hi in zip(self.lower, self.upper, strict=True)):
            raise SelectionError(f"lower {self.lower} exceeds upper {self.upper}")

    def _mask(self, coords: _F64, mesh: Mesh) -> _Bool:
        _check_dim(self.lower, coords)
        t = _tol(self.tol, mesh)
        lo = np.asarray(self.lower) - t
        hi = np.asarray(self.upper) + t
        inside = ((coords >= lo) & (coords <= hi)).all(axis=1)
        return np.asarray(inside, dtype=bool)


@dataclass(frozen=True)
class Sphere(SelectorBase):
    """Ball of ``radius`` around ``center`` (a disk in 2D)."""

    center: tuple[float, ...]
    radius: float
    tol: float | None = None

    def __post_init__(self) -> None:
        _freeze(
            self,
            center=tuple(float(x) for x in self.center),
            radius=float(self.radius),
            tol=None if self.tol is None else float(self.tol),
        )
        if self.radius < 0.0:
            raise SelectionError(f"radius must be >= 0, got {self.radius}")

    def _mask(self, coords: _F64, mesh: Mesh) -> _Bool:
        _check_dim(self.center, coords)
        dist = np.linalg.norm(coords - np.asarray(self.center), axis=1)
        out: _Bool = dist <= self.radius + _tol(self.tol, mesh)
        return out


@dataclass(frozen=True)
class Cylinder(SelectorBase):
    """Finite cylinder around the axis segment ``p0``-``p1``."""

    p0: tuple[float, ...]
    p1: tuple[float, ...]
    radius: float
    tol: float | None = None

    def __post_init__(self) -> None:
        _freeze(
            self,
            p0=tuple(float(x) for x in self.p0),
            p1=tuple(float(x) for x in self.p1),
            radius=float(self.radius),
            tol=None if self.tol is None else float(self.tol),
        )
        if len(self.p0) != len(self.p1):
            raise SelectionError("p0 and p1 must have the same length")
        if not np.linalg.norm(np.asarray(self.p1) - np.asarray(self.p0)) > 0.0:
            raise SelectionError("cylinder axis has zero length")
        if self.radius < 0.0:
            raise SelectionError(f"radius must be >= 0, got {self.radius}")

    def _mask(self, coords: _F64, mesh: Mesh) -> _Bool:
        _check_dim(self.p0, coords)
        t = _tol(self.tol, mesh)
        p0 = np.asarray(self.p0)
        axis = np.asarray(self.p1) - p0
        length = float(np.linalg.norm(axis))
        unit = axis / length
        rel = coords - p0
        along = rel @ unit
        radial = np.linalg.norm(rel - along[:, None] * unit, axis=1)
        out: _Bool = (along >= -t) & (along <= length + t) & (radial <= self.radius + t)
        return out


@dataclass(frozen=True)
class PlaneSlab(SelectorBase):
    """Points within ``tol`` of the plane through ``point`` with ``normal``."""

    point: tuple[float, ...]
    normal: tuple[float, ...]
    tol: float | None = None

    def __post_init__(self) -> None:
        _freeze(
            self,
            point=tuple(float(x) for x in self.point),
            normal=tuple(float(x) for x in self.normal),
            tol=None if self.tol is None else float(self.tol),
        )
        if len(self.point) != len(self.normal):
            raise SelectionError("point and normal must have the same length")
        if not np.linalg.norm(np.asarray(self.normal)) > 0.0:
            raise SelectionError("normal has zero length")

    def _mask(self, coords: _F64, mesh: Mesh) -> _Bool:
        _check_dim(self.point, coords)
        n = np.asarray(self.normal)
        n = n / np.linalg.norm(n)
        dist = np.abs((coords - np.asarray(self.point)) @ n)
        out: _Bool = dist <= _tol(self.tol, mesh)
        return out


@dataclass(frozen=True)
class Predicate(SelectorBase):
    """Custom vectorized membership test ``fn(coords (n, dim)) -> bool (n,)``.

    Not expressible in the declarative schema. A lambda makes the selector
    unpicklable; use a module-level function if the study crosses processes.
    """

    fn: Callable[[_F64], npt.NDArray[np.bool_]]

    def _mask(self, coords: _F64, mesh: Mesh) -> _Bool:
        out = np.asarray(self.fn(coords))
        if out.shape != (coords.shape[0],) or out.dtype != np.bool_:
            raise SelectionError(
                f"predicate must return bool of shape ({coords.shape[0]},), "
                f"got {out.dtype} {out.shape}"
            )
        return out


@dataclass(frozen=True)
class NearPoint(SelectorBase):
    """The ``k`` entities nearest to ``point``; ties break to the lowest id."""

    point: tuple[float, ...]
    k: int = 1

    def __post_init__(self) -> None:
        _freeze(self, point=tuple(float(x) for x in self.point), k=int(self.k))
        if self.k < 1:
            raise SelectionError(f"k must be >= 1, got {self.k}")

    def nodes(self, mesh: Mesh) -> _I64:
        """Sorted ids of the k nearest nodes."""
        return self._nearest(mesh.nodes)

    def elements(self, mesh: Mesh) -> _I64:
        """Sorted ids of the k nearest element centroids."""
        return self._nearest(mesh.element_centroids)

    def faces(self, mesh: Mesh) -> _I64:
        """Sorted ids of the k nearest boundary-face centroids."""
        return self._nearest(mesh.boundary_faces().centroid)

    def _nearest(self, coords: _F64) -> _I64:
        _check_dim(self.point, coords)
        dist = np.linalg.norm(coords - np.asarray(self.point), axis=1)
        order = np.argsort(dist, kind="stable")[: self.k]
        return np.sort(order).astype(np.int64)


@dataclass(frozen=True)
class OnBoundary(SelectorBase):
    """Everything on the boundary of the active domain."""

    def nodes(self, mesh: Mesh) -> _I64:
        """Return the unique nodes of all boundary faces."""
        return np.unique(mesh.boundary_faces().nodes).astype(np.int64)

    def elements(self, mesh: Mesh) -> _I64:
        """Return the unique owner elements of all boundary faces."""
        return np.unique(mesh.boundary_faces().owner).astype(np.int64)

    def faces(self, mesh: Mesh) -> _I64:
        """All boundary-face ids."""
        return np.arange(mesh.boundary_faces().n_faces, dtype=np.int64)


@dataclass(frozen=True)
class FaceSetSelector(SelectorBase):
    """Explicit boundary-face ids; the target of the CAD face bridge."""

    face_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _freeze(self, face_ids=tuple(int(x) for x in self.face_ids))

    def faces(self, mesh: Mesh) -> _I64:
        """Return the validated face ids, sorted unique."""
        n = mesh.boundary_faces().n_faces
        ids = np.unique(np.asarray(self.face_ids, dtype=np.int64))
        if ids.size and (ids[0] < 0 or ids[-1] >= n):
            raise SelectionError(f"face id out of range [0, {n}): {self.face_ids}")
        return ids

    def nodes(self, mesh: Mesh) -> _I64:
        """Return the unique nodes of the selected faces."""
        return np.unique(mesh.boundary_faces().nodes[self.faces(mesh)]).astype(np.int64)

    def elements(self, mesh: Mesh) -> _I64:
        """Return the unique owner elements of the selected faces."""
        return np.unique(mesh.boundary_faces().owner[self.faces(mesh)]).astype(np.int64)


@dataclass(frozen=True)
class _And(SelectorBase):
    a: Selector
    b: Selector

    def __repr__(self) -> str:
        return f"({self.a!r} & {self.b!r})"

    def nodes(self, mesh: Mesh) -> _I64:
        return np.intersect1d(self.a.nodes(mesh), self.b.nodes(mesh))

    def elements(self, mesh: Mesh) -> _I64:
        return np.intersect1d(self.a.elements(mesh), self.b.elements(mesh))

    def faces(self, mesh: Mesh) -> _I64:
        return np.intersect1d(self.a.faces(mesh), self.b.faces(mesh))


@dataclass(frozen=True)
class _Or(SelectorBase):
    a: Selector
    b: Selector

    def __repr__(self) -> str:
        return f"({self.a!r} | {self.b!r})"

    def nodes(self, mesh: Mesh) -> _I64:
        return np.union1d(self.a.nodes(mesh), self.b.nodes(mesh))

    def elements(self, mesh: Mesh) -> _I64:
        return np.union1d(self.a.elements(mesh), self.b.elements(mesh))

    def faces(self, mesh: Mesh) -> _I64:
        return np.union1d(self.a.faces(mesh), self.b.faces(mesh))


@dataclass(frozen=True)
class _Not(SelectorBase):
    inner: Selector

    def __repr__(self) -> str:
        return f"~{self.inner!r}"

    def nodes(self, mesh: Mesh) -> _I64:
        return np.setdiff1d(np.arange(mesh.n_nodes, dtype=np.int64), self.inner.nodes(mesh))

    def elements(self, mesh: Mesh) -> _I64:
        return np.setdiff1d(np.arange(mesh.n_elements, dtype=np.int64), self.inner.elements(mesh))

    def faces(self, mesh: Mesh) -> _I64:
        universe = np.arange(mesh.boundary_faces().n_faces, dtype=np.int64)
        return np.setdiff1d(universe, self.inner.faces(mesh))
