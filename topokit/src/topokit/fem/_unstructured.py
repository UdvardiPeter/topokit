# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Linear elasticity on unstructured simplex meshes.

``UnstructuredElasticity`` implements the ``PhysicsModel`` protocol, mirroring
``LinearElasticity``'s construction and per-iteration contract but over a
per-element stiffness *stack* (``fem._elements.stiffness_stack``) instead of
a single shared element matrix, since simplex elements are not geometrically
identical across a mesh. Supports, loads, condensation and rigid-body modes
reuse the mesh-generic machinery in :mod:`topokit.fem`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from topokit.backend import SparseMatrix, active_backend, get_kernel, register_kernel
from topokit.fem import (
    FemError,
    Load,
    Material,
    _build_loads,
    _csr_pattern_positions,
    _d_matrix,
    _elements,
    _fixed_dofs,
    _rigid_body_modes,
    _von_mises,
)
from topokit.fields import FieldSpec, NodeField
from topokit.mesh import UnstructuredMesh
from topokit.selection import Selector

_F64 = npt.NDArray[np.float64]
_I64 = npt.NDArray[np.int64]


def _assemble_csr_data_stack(scale_active: Any, ke_stack_flat: Any, pos: Any, nnz: int) -> _F64:
    """Fill CSR data from a per-element stiffness stack.

    Mirrors ``fem._assemble_csr_data``'s scatter-per-local-pair loop and
    ``pos``/sentinel conventions exactly, replacing the shared scalar
    ``ke[k]`` with the per-active-element vector ``ke[:, k]``. Coerces
    inputs to host numpy so it stays correct (if slow) as the fallback for
    device-array backends.
    """
    sa = np.asarray(scale_active, dtype=np.float64)
    ke = np.asarray(ke_stack_flat, dtype=np.float64).reshape(sa.shape[0], -1)
    p = np.asarray(pos)
    data = np.zeros(nnz + 1)
    for k in range(p.shape[0]):
        np.add.at(data, p[k], ke[:, k] * sa)
    return data[:-1]


register_kernel("assemble_csr_data_stack", "generic", _assemble_csr_data_stack)


class UnstructuredElasticity:
    """Linear static elasticity on a simplex mesh (tri3, tet4, or tet10)."""

    expected_field: ClassVar[FieldSpec] = FieldSpec("stiffness_scale")

    def __init__(
        self,
        mesh: UnstructuredMesh,
        material: Material,
        supports: Sequence[tuple[Selector, str | Sequence[str | int]]],
        loads: Sequence[Load] | Sequence[Sequence[Load]],
        mode: str = "plane_stress",
    ) -> None:
        self.mesh = mesh
        self.material = material
        self.mode = mode
        dim = mesh.dim
        if dim == 3 and mode != "plane_stress":
            raise FemError("mode is a 2D setting; remove it for 3D meshes")
        kind = mesh.element_kind
        if kind not in _elements.NDOF:
            raise FemError(
                f"unsupported element kind {kind!r}; supported: {sorted(_elements.NDOF)}"
            )

        self._d = _d_matrix(material, dim, mode)
        conn = mesh.element_nodes[mesh.active_elements]
        self._ke_stack = _elements.stiffness_stack(kind, self._d, mesh.nodes, conn)
        self._ke_stack.flags.writeable = False
        self._centroid_b_stack = _elements.centroid_b_stack(kind, mesh.nodes, conn)
        self._centroid_b_stack.flags.writeable = False

        # condensed DOF numbering over active nodes
        cn = mesh.node_index_map
        n_cdof = int(mesh.active_nodes.sum()) * dim

        fixed = _fixed_dofs(mesh, supports, cn, dim, n_cdof)
        free = ~fixed
        self._free_index = np.full(n_cdof, -1, dtype=np.int64)
        self._free_index[free] = np.arange(int(free.sum()), dtype=np.int64)
        self._n_dof = int(free.sum())
        if self._n_dof == 0:
            raise FemError("all DOFs are fixed")

        # element DOF pattern over active elements, mapped to free numbering
        nen = conn.shape[1]
        cdofs = (cn[conn][:, :, None] * dim + np.arange(dim)).reshape(-1, nen * dim)
        self._edofs_free = self._free_index[cdofs]  # (n_ae, nen*dim), -1 where fixed
        self._ke_stack_flat = self._ke_stack.reshape(self._ke_stack.shape[0], -1)
        self._indptr, self._indices, self._pos = _csr_pattern_positions(
            self._edofs_free.astype(np.int64), self._n_dof
        )
        self._nnz = int(self._indices.size)

        face_w = _elements.face_weights(kind)
        body_w = _elements.body_weights(kind)
        self._loads = _build_loads(
            mesh, loads, cn, dim, n_cdof, free, lambda _n: face_w, lambda _n: body_w
        )
        self._loads.flags.writeable = False

    @property
    def n_dof(self) -> int:
        """Number of free DOFs."""
        return self._n_dof

    @property
    def n_cases(self) -> int:
        """Number of load cases."""
        return int(self._loads.shape[1])

    def dof_index(self, node: int, comp: int) -> int:
        """Free-DOF index for ``(node, component)``; ``-1`` if fixed or inactive."""
        if comp < 0 or comp >= self.mesh.dim:
            raise FemError(f"component {comp} out of range for dim {self.mesh.dim}")
        cnid = int(self.mesh.node_index_map[node])
        if cnid < 0:
            return -1
        return int(self._free_index[cnid * self.mesh.dim + comp])

    def near_nullspace(self) -> _F64:
        """Rigid-body modes on the free DOFs, shape ``(n_dof, 3 in 2D / 6 in 3D)``.

        The near-nullspace hint for AMG solvers: translations plus rotations
        about the active-node centroid. Supports truncate the modes; that is
        expected, the hint only needs to span the low-energy space.
        """
        return _rigid_body_modes(self.mesh, self._free_index, self.mesh.dim)

    def _check_scale(self, scale: Any) -> _F64:
        arr = np.asarray(scale, dtype=np.float64)
        if arr.shape != (self.mesh.n_elements,):
            raise FemError(f"scale shape {arr.shape} != ({self.mesh.n_elements},)")
        if not np.isfinite(arr).all():
            raise FemError("scale contains non-finite values")
        if bool((arr < 0.0).any()):
            raise FemError("scale contains negative values")
        return arr

    def assemble(self, scale: Any) -> SparseMatrix:
        """Free-DOF stiffness matrix for per-element ``scale`` multipliers."""
        arr = self._check_scale(scale)
        sa = arr[self.mesh.active_elements]
        backend = active_backend()
        kernel = get_kernel("assemble_csr_data_stack")
        data = kernel(backend.asarray(sa), self._ke_stack_flat, self._pos, self._nnz)
        return backend.csr_from_parts(
            data, self._indices, self._indptr, shape=(self._n_dof, self._n_dof)
        )

    def loads(self) -> _F64:
        """Free-DOF load matrix, shape ``(n_dof, n_cases)``."""
        return self._loads

    def _element_displacements(self, u: Any) -> _F64:
        arr = np.asarray(u, dtype=np.float64)
        if arr.shape != (self._n_dof,):
            raise FemError(f"u shape {arr.shape} != ({self._n_dof},)")
        u_ext = np.append(arr, 0.0)
        idx = np.where(self._edofs_free >= 0, self._edofs_free, self._n_dof)
        return u_ext[idx]

    def element_energies(self, u: Any, scale: Any) -> _F64:
        """Per-element strain energy ``scale_e * u_e K_e u_e``, zeros at void.

        Sums to the compliance ``u f``. Pass ones as ``scale`` to get the
        unscaled energies ``u_e K_e u_e``, the compliance sensitivity kernel.
        """
        arr = self._check_scale(scale)
        ue = self._element_displacements(u)
        e_active = np.einsum("ei,eij,ej->e", ue, self._ke_stack, ue)
        out = np.zeros(self.mesh.n_elements)
        out[self.mesh.active_elements] = e_active * arr[self.mesh.active_elements]
        return out

    def element_stress(self, u: Any, scale: Any) -> _F64:
        """Centroid von Mises stress per element, zeros at void."""
        arr = self._check_scale(scale)
        ue = self._element_displacements(u)
        strain = np.einsum("eij,ej->ei", self._centroid_b_stack, ue)
        stress = strain @ self._d.T
        stress *= arr[self.mesh.active_elements][:, None]
        vm = _von_mises(stress, self.mesh.dim, self.mode, self.material.nu)
        out = np.zeros(self.mesh.n_elements)
        out[self.mesh.active_elements] = vm
        return out

    def displacement_field(self, u: Any, name: str = "u") -> NodeField:
        """Expand a free-DOF vector to a full nodal field (zeros at fixed/inactive)."""
        arr = np.asarray(u, dtype=np.float64)
        if arr.shape != (self._n_dof,):
            raise FemError(f"u shape {arr.shape} != ({self._n_dof},)")
        dim = self.mesh.dim
        full = np.zeros((self.mesh.n_nodes, dim))
        cn = self.mesh.node_index_map
        active = np.flatnonzero(cn >= 0)
        for comp in range(dim):
            fidx = self._free_index[cn[active] * dim + comp]
            sel = fidx >= 0
            full[active[sel], comp] = arr[fidx[sel]]
        return NodeField(full, self.mesh, name=name)
