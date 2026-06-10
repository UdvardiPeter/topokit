# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Typed field containers binding arrays to their mesh.

Construction validates shape and coerces to float64. ``save`` and ``load``
round-trip through npz. The mesh is referenced through the structural
``MeshLike`` protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Self, runtime_checkable

import numpy as np
import numpy.typing as npt


@runtime_checkable
class MeshLike(Protocol):
    """The minimal mesh interface fields need."""

    @property
    def n_nodes(self) -> int:
        """Number of mesh nodes."""
        ...

    @property
    def n_elements(self) -> int:
        """Number of mesh elements."""
        ...

    @property
    def dim(self) -> int:
        """Spatial dimension, 2 or 3."""
        ...


class FieldError(ValueError):
    """Shape, dtype, or mesh mismatch."""


class _Field:
    """Shared container behavior. Subclasses define the expected shape."""

    def __init__(self, values: npt.ArrayLike, mesh: MeshLike, *, name: str) -> None:
        arr = np.array(values, dtype=np.float64, copy=True)
        expected = self._expected_shape(mesh)
        if arr.shape != expected:
            raise FieldError(
                f"{type(self).__name__} {name!r}: shape {arr.shape} != expected {expected} "
                f"(mesh: {mesh.n_elements} elements, {mesh.n_nodes} nodes)"
            )
        arr.flags.writeable = False
        self.values: npt.NDArray[np.float64] = arr
        self.mesh = mesh
        self.name = name

    def _expected_shape(self, mesh: MeshLike) -> tuple[int, ...]:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        """Write values and mesh identity to ``path`` as npz."""
        np.savez(
            path,
            values=self.values,
            name=np.str_(self.name),
            n_elements=self.mesh.n_elements,
            n_nodes=self.mesh.n_nodes,
        )

    @classmethod
    def load(cls, path: str | Path, mesh: MeshLike) -> Self:
        """Read a field written by :meth:`save`, validating it matches ``mesh``."""
        with np.load(path) as data:
            saved_el = int(data["n_elements"])
            saved_no = int(data["n_nodes"])
            if saved_el != mesh.n_elements or saved_no != mesh.n_nodes:
                raise FieldError(
                    f"{path}: saved for mesh with {saved_el} elements / {saved_no} nodes, "
                    f"given mesh has {mesh.n_elements} / {mesh.n_nodes}"
                )
            return cls(data["values"], mesh, name=str(data["name"]))


class DesignField(_Field):
    """Element-wise design density, shape ``(n_elements,)``."""

    def _expected_shape(self, mesh: MeshLike) -> tuple[int, ...]:
        return (mesh.n_elements,)


class ElementField(_Field):
    """Element-wise scalar field such as von Mises stress, shape ``(n_elements,)``."""

    def _expected_shape(self, mesh: MeshLike) -> tuple[int, ...]:
        return (mesh.n_elements,)


class NodeField(_Field):
    """Nodal vector field such as displacement, shape ``(n_nodes, dim)``."""

    def _expected_shape(self, mesh: MeshLike) -> tuple[int, ...]:
        return (mesh.n_nodes, mesh.dim)
