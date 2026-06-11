# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Linear elasticity on structured grids.

``LinearElasticity`` implements the ``PhysicsModel`` protocol: it assembles
the free-DOF stiffness system for a given per-element stiffness scale (the
parametrization chain's output), builds consistent load vectors, and
evaluates element energies and von Mises stresses.

Implementation notes:

- All elements of a structured grid are geometrically identical, so one
  element stiffness matrix is computed per model and assembly is a
  scale-multiply into a precomputed COO pattern.
- Boundary conditions use reduction: fixed DOFs are dropped from the
  pattern before CSR conversion. Supports are zero-valued (homogeneous
  Dirichlet) in v1.
- Void elements are excluded from the system through the mesh's
  condensation map. Setup-time arrays are host numpy; the per-iteration
  ``assemble`` path goes through the array backend.
- 2D uses unit thickness; ``mode`` selects plane stress (default) or
  plane strain.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from topokit.backend import ArrayBackend, SparseMatrix, default_backend
from topokit.fields import FieldSpec, NodeField
from topokit.mesh import StructuredGrid
from topokit.selection import Selector

_F64 = npt.NDArray[np.float64]
_I64 = npt.NDArray[np.int64]

_GP = 1.0 / np.sqrt(3.0)


class FemError(ValueError):
    """Invalid physics setup or input."""


@dataclass(frozen=True)
class Material:
    """Isotropic linear-elastic material.

    Unit-agnostic; the built-in table uses mm-N-MPa-tonne units.
    """

    E: float
    nu: float
    rho: float
    name: str = "custom"

    def __post_init__(self) -> None:
        if self.E <= 0.0:
            raise FemError(f"E must be > 0, got {self.E}")
        if not -1.0 < self.nu < 0.5:
            raise FemError(f"nu must be in (-1, 0.5), got {self.nu}")


# Representative handbook values in mm-N-MPa-tonne units; verify against
# supplier data before using for real parts.
STEEL = Material(E=210000.0, nu=0.30, rho=7.85e-9, name="steel")
ALUMINUM_6061 = Material(E=68900.0, nu=0.33, rho=2.70e-9, name="aluminum-6061")
ABS = Material(E=2300.0, nu=0.35, rho=1.05e-9, name="abs")
PA12 = Material(E=1700.0, nu=0.40, rho=1.01e-9, name="pa12")
RESIN_SLA = Material(E=2800.0, nu=0.35, rho=1.15e-9, name="resin-sla")


@dataclass(frozen=True)
class PointLoad:
    """Total ``force`` split equally over the selected active nodes."""

    selector: Selector
    force: tuple[float, ...]


@dataclass(frozen=True)
class SurfaceTraction:
    """Per-area load on selected boundary faces.

    Give exactly one of ``traction`` (a vector) or ``pressure`` (a scalar;
    positive pushes against the outward normal).
    """

    selector: Selector
    traction: tuple[float, ...] | None = None
    pressure: float | None = None

    def __post_init__(self) -> None:
        if (self.traction is None) == (self.pressure is None):
            raise FemError("give exactly one of traction or pressure")


@dataclass(frozen=True)
class BodyForce:
    """Per-volume force over all active elements."""

    vector: tuple[float, ...]


Load = PointLoad | SurfaceTraction | BodyForce


@runtime_checkable
class PhysicsModel(Protocol):
    """The physics contract the orchestration layer is written against."""

    expected_field: ClassVar[FieldSpec]

    @property
    def n_dof(self) -> int:
        """Number of free DOFs."""
        ...

    @property
    def n_cases(self) -> int:
        """Number of load cases."""
        ...

    def assemble(self, scale: Any) -> SparseMatrix:
        """Free-DOF stiffness matrix for per-element ``scale`` multipliers."""
        ...

    def loads(self) -> Any:
        """Free-DOF load matrix, shape ``(n_dof, n_cases)``."""
        ...

    def element_energies(self, u: Any, scale: Any) -> Any:
        """Per-element strain energy, shape ``(n_elements,)``, zeros at void."""
        ...

    def element_stress(self, u: Any, scale: Any) -> Any:
        """Per-element von Mises stress, shape ``(n_elements,)``, zeros at void."""
        ...


def _d_matrix(material: Material, dim: int, mode: str) -> _F64:
    e, nu = material.E, material.nu
    if dim == 2:
        if mode == "plane_stress":
            c = e / (1.0 - nu * nu)
            return np.array([[c, c * nu, 0.0], [c * nu, c, 0.0], [0.0, 0.0, c * (1.0 - nu) / 2.0]])
        if mode == "plane_strain":
            c = e / ((1.0 + nu) * (1.0 - 2.0 * nu))
            return np.array(
                [
                    [c * (1.0 - nu), c * nu, 0.0],
                    [c * nu, c * (1.0 - nu), 0.0],
                    [0.0, 0.0, c * (1.0 - 2.0 * nu) / 2.0],
                ]
            )
        raise FemError(f"unknown 2D mode {mode!r}")
    lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = e / (2.0 * (1.0 + nu))
    d = np.zeros((6, 6))
    d[:3, :3] = lam
    d[np.arange(3), np.arange(3)] = lam + 2.0 * mu
    d[np.arange(3, 6), np.arange(3, 6)] = mu
    return d


_CORNERS_2D = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
_CORNERS_3D = np.array(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ]
)


def _shape_gradients(xi: _F64, corners: _F64, spacing: tuple[float, ...]) -> _F64:
    """Physical shape-function gradients at local point ``xi``, shape (nen, dim).

    ``dN_i/dxi_a = (corner_ia / 2**dim) * prod_{b != a} (1 + xi_b * corner_ib)``,
    mapped to physical coordinates by the rectangular Jacobian ``2 / h_a``.
    """
    nen, dim = corners.shape
    grads = np.empty((nen, dim))
    for i in range(nen):
        for a in range(dim):
            term = corners[i, a] / (2.0**dim)
            for b in range(dim):
                if b != a:
                    term *= 1.0 + xi[b] * corners[i, b]
            grads[i, a] = term * 2.0 / spacing[a]
    return grads


def _b_matrix(grads: _F64) -> _F64:
    """Strain-displacement matrix from physical gradients, node-major DOFs."""
    nen, dim = grads.shape
    if dim == 2:
        b = np.zeros((3, 2 * nen))
        b[0, 0::2] = grads[:, 0]
        b[1, 1::2] = grads[:, 1]
        b[2, 0::2] = grads[:, 1]
        b[2, 1::2] = grads[:, 0]
        return b
    b = np.zeros((6, 3 * nen))
    b[0, 0::3] = grads[:, 0]
    b[1, 1::3] = grads[:, 1]
    b[2, 2::3] = grads[:, 2]
    b[3, 0::3] = grads[:, 1]
    b[3, 1::3] = grads[:, 0]
    b[4, 1::3] = grads[:, 2]
    b[4, 2::3] = grads[:, 1]
    b[5, 0::3] = grads[:, 2]
    b[5, 2::3] = grads[:, 0]
    return b


def _element_stiffness(d: _F64, spacing: tuple[float, ...]) -> _F64:
    dim = len(spacing)
    corners = _CORNERS_2D if dim == 2 else _CORNERS_3D
    det_j = float(np.prod(spacing)) / (2.0**dim)
    nen = corners.shape[0]
    ke = np.zeros((nen * dim, nen * dim))
    for gp in corners * _GP:  # full Gauss points share the corner sign pattern
        b = _b_matrix(_shape_gradients(gp, corners, spacing))
        ke += b.T @ d @ b * det_j
    return ke


def _centroid_b(spacing: tuple[float, ...]) -> _F64:
    dim = len(spacing)
    corners = _CORNERS_2D if dim == 2 else _CORNERS_3D
    return _b_matrix(_shape_gradients(np.zeros(dim), corners, spacing))


def _von_mises(stress: _F64, dim: int, mode: str, nu: float) -> _F64:
    if dim == 3:
        sx, sy, sz, txy, tyz, tzx = stress.T
        out = np.sqrt(
            0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
            + 3.0 * (txy**2 + tyz**2 + tzx**2)
        )
        return np.asarray(out, dtype=np.float64)
    sx, sy, txy = stress.T
    if mode == "plane_strain":
        sz = nu * (sx + sy)
        out = np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3.0 * txy**2)
        return np.asarray(out, dtype=np.float64)
    return np.asarray(np.sqrt(sx**2 - sx * sy + sy**2 + 3.0 * txy**2), dtype=np.float64)


def _parse_comps(dofs: str | Sequence[str | int], dim: int) -> tuple[int, ...]:
    names = {"x": 0, "y": 1, "z": 2}
    if isinstance(dofs, str):
        if dofs == "all":
            return tuple(range(dim))
        if dofs in names:
            comps: tuple[int, ...] = (names[dofs],)
        else:
            raise FemError(f"unknown dof spec {dofs!r}")
    else:
        comps = tuple(names[c] if isinstance(c, str) else int(c) for c in dofs)
    if any(c < 0 or c >= dim for c in comps):
        raise FemError(f"dof components {comps} out of range for dim {dim}")
    return comps


class LinearElasticity:
    """Linear static elasticity on a structured grid."""

    expected_field: ClassVar[FieldSpec] = FieldSpec("stiffness_scale")

    def __init__(
        self,
        mesh: StructuredGrid,
        material: Material,
        supports: Sequence[tuple[Selector, str | Sequence[str | int]]],
        loads: Sequence[Load] | Sequence[Sequence[Load]],
        mode: str = "plane_stress",
        backend: ArrayBackend | None = None,
    ) -> None:
        self.mesh = mesh
        self.material = material
        self.mode = mode
        self._backend = backend if backend is not None else default_backend()
        dim = mesh.dim

        self._d = _d_matrix(material, dim, mode)
        self.element_stiffness = _element_stiffness(self._d, mesh.spacing)
        self._centroid_b = _centroid_b(mesh.spacing)

        # condensed DOF numbering over active nodes
        cn = mesh.node_index_map
        n_cdof = int(mesh.active_nodes.sum()) * dim

        fixed = np.zeros(n_cdof, dtype=bool)
        if not supports:
            raise FemError("at least one support is required")
        for selector, dofs in supports:
            nids = selector.nodes(mesh)
            cnids = cn[nids]
            cnids = cnids[cnids >= 0]
            if cnids.size == 0:
                raise FemError(f"support selector {selector!r} selects no active nodes")
            for comp in _parse_comps(dofs, dim):
                fixed[cnids * dim + comp] = True
        free = ~fixed
        self._free_index = np.full(n_cdof, -1, dtype=np.int64)
        self._free_index[free] = np.arange(int(free.sum()), dtype=np.int64)
        self._n_dof = int(free.sum())
        if self._n_dof == 0:
            raise FemError("all DOFs are fixed")

        # element DOF pattern over active elements, mapped to free numbering
        conn = mesh.element_nodes[mesh.active_elements]
        nen = conn.shape[1]
        cdofs = (cn[conn][:, :, None] * dim + np.arange(dim)).reshape(-1, nen * dim)
        self._edofs_free = self._free_index[cdofs]  # (n_ae, nen*dim), -1 where fixed
        rows = np.repeat(self._edofs_free, nen * dim, axis=1).ravel()
        cols = np.tile(self._edofs_free, (1, nen * dim)).ravel()
        keep = (rows >= 0) & (cols >= 0)
        self._rows = rows[keep]
        self._cols = cols[keep]
        self._keep = keep
        self._ke_flat = self.element_stiffness.ravel()

        self._loads = self._build_loads(loads, cn, dim, n_cdof, free)

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
        cnid = int(self.mesh.node_index_map[node])
        if cnid < 0:
            return -1
        return int(self._free_index[cnid * self.mesh.dim + comp])

    def _build_loads(
        self,
        loads: Sequence[Load] | Sequence[Sequence[Load]],
        cn: _I64,
        dim: int,
        n_cdof: int,
        free: npt.NDArray[np.bool_],
    ) -> _F64:
        if not loads:
            raise FemError("at least one load is required")
        cases: list[Sequence[Load]]
        if isinstance(loads[0], PointLoad | SurfaceTraction | BodyForce):
            cases = [loads]  # type: ignore[list-item]
        else:
            cases = list(loads)  # type: ignore[arg-type]
        out = np.zeros((n_cdof, len(cases)))
        for j, case in enumerate(cases):
            for load in case:
                self._apply_load(out[:, j], load, cn, dim)
        return out[free, :]

    def _apply_load(self, f: _F64, load: Load, cn: _I64, dim: int) -> None:
        mesh = self.mesh
        if isinstance(load, PointLoad):
            if len(load.force) != dim:
                raise FemError(f"force dim {len(load.force)} != mesh dim {dim}")
            nids = load.selector.nodes(mesh)
            cnids = cn[nids]
            cnids = cnids[cnids >= 0]
            if cnids.size == 0:
                raise FemError(f"point load selector {load.selector!r} selects no active nodes")
            share = np.asarray(load.force) / cnids.size
            for comp in range(dim):
                np.add.at(f, cnids * dim + comp, share[comp])
            return
        if isinstance(load, SurfaceTraction):
            fids = load.selector.faces(mesh)
            if fids.size == 0:
                raise FemError(f"traction selector {load.selector!r} selects no boundary faces")
            bf = mesh.boundary_faces()
            if load.traction is not None:
                if len(load.traction) != dim:
                    raise FemError(f"traction dim {len(load.traction)} != mesh dim {dim}")
                t = np.tile(np.asarray(load.traction), (fids.size, 1))
            else:
                assert load.pressure is not None
                t = -load.pressure * bf.normal[fids]
            face_nodes = bf.nodes[fids]
            nodes_per_face = face_nodes.shape[1]
            weights = bf.area[fids][:, None] / nodes_per_face
            contrib = t * weights  # (n_f, dim) per node
            cnids = cn[face_nodes]
            for comp in range(dim):
                np.add.at(
                    f, (cnids * dim + comp).ravel(), np.repeat(contrib[:, comp], nodes_per_face)
                )
            return
        if len(load.vector) != dim:
            raise FemError(f"body force dim {len(load.vector)} != mesh dim {dim}")
        conn = mesh.element_nodes[mesh.active_elements]
        nen = conn.shape[1]
        per_node = (
            np.asarray(load.vector)[None, :]
            * mesh.element_volumes[mesh.active_elements][:, None]
            / nen
        )
        cnids = cn[conn]
        for comp in range(dim):
            np.add.at(f, (cnids * dim + comp).ravel(), np.repeat(per_node[:, comp], nen))

    def _check_scale(self, scale: Any) -> _F64:
        arr = np.asarray(scale, dtype=np.float64)
        if arr.shape != (self.mesh.n_elements,):
            raise FemError(f"scale shape {arr.shape} != ({self.mesh.n_elements},)")
        if not np.isfinite(arr).all():
            raise FemError("scale contains non-finite values")
        return arr

    def assemble(self, scale: Any) -> SparseMatrix:
        """Free-DOF stiffness matrix for per-element ``scale`` multipliers."""
        arr = self._check_scale(scale)
        bk = self._backend
        sa = arr[self.mesh.active_elements]
        vals = bk.einsum("e,k->ek", bk.asarray(sa), bk.asarray(self._ke_flat)).ravel()
        return bk.coo_to_csr(
            self._rows, self._cols, vals[self._keep], shape=(self._n_dof, self._n_dof)
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
        """Per-element strain energy ``scale_e * u_e K_e u_e``, zeros at void."""
        arr = self._check_scale(scale)
        ue = self._element_displacements(u)
        e_active = np.einsum("ei,ij,ej->e", ue, self.element_stiffness, ue)
        out = np.zeros(self.mesh.n_elements)
        out[self.mesh.active_elements] = e_active * arr[self.mesh.active_elements]
        return out

    def element_stress(self, u: Any, scale: Any) -> _F64:
        """Centroid von Mises stress per element, zeros at void."""
        arr = self._check_scale(scale)
        ue = self._element_displacements(u)
        stress = ue @ self._centroid_b.T @ self._d.T
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
