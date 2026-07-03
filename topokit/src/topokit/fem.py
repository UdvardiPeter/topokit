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
- Bilinear elements with full integration lock in coarse bending-dominated
  meshes; benchmark meshes follow literature norms.
- Models are not meant to cross processes: they do not keep their
  constructor inputs, so reconstruct from the problem spec instead of
  pickling.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from topokit.backend import SparseMatrix, active_backend, get_kernel, register_kernel
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


def _freeze(obj: object, **fields: object) -> None:
    for name, value in fields.items():
        object.__setattr__(obj, name, value)


@dataclass(frozen=True)
class PointLoad:
    """Total ``force`` split equally over the selected active nodes."""

    selector: Selector
    force: tuple[float, ...]

    def __post_init__(self) -> None:
        _freeze(self, force=tuple(float(x) for x in self.force))


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
        _freeze(
            self,
            traction=None if self.traction is None else tuple(float(x) for x in self.traction),
            pressure=None if self.pressure is None else float(self.pressure),
        )


@dataclass(frozen=True)
class BodyForce:
    """Per-volume force over all active elements.

    Design-independent: evaluated once at full density. Density-dependent
    self-weight (loads that shrink with the material) is out of scope in v1.
    """

    vector: tuple[float, ...]

    def __post_init__(self) -> None:
        _freeze(self, vector=tuple(float(x) for x in self.vector))


Load = PointLoad | SurfaceTraction | BodyForce


@runtime_checkable
class PhysicsModel(Protocol):
    """The physics contract the orchestration layer is written against."""

    expected_field: ClassVar[FieldSpec]

    @property
    def mesh(self) -> StructuredGrid:
        """The discretization the model is built on."""
        ...

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


def _csr_pattern_positions(
    edofs_free: _I64, n_dof: int, chunk: int = 16384
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """CSR pattern and per-entry data positions for a fixed element DOF map.

    Returns ``(indptr, indices, pos)`` with ``pos[k, e]`` the CSR data slot of
    element ``e``'s local stiffness entry ``k = a*m + b``; entries touching a
    fixed DOF point at the trailing dump slot ``nnz``. Chunked so setup
    transients stay bounded.
    """
    n_ae, m = edofs_free.shape
    keys_unique = np.empty(0, dtype=np.int64)
    for lo in range(0, n_ae, chunk):
        ef = edofs_free[lo : lo + chunk].astype(np.int64)
        rows = np.repeat(ef, m, axis=1).ravel()
        cols = np.tile(ef, (1, m)).ravel()
        keep = (rows >= 0) & (cols >= 0)
        keys_unique = np.union1d(keys_unique, rows[keep] * n_dof + cols[keep])
    nnz = int(keys_unique.size)
    indices = (keys_unique % n_dof).astype(np.int32)
    counts = np.bincount((keys_unique // n_dof).astype(np.int64), minlength=n_dof)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    pos = np.full((m * m, n_ae), nnz, dtype=np.int32)
    for lo in range(0, n_ae, chunk):
        ef = edofs_free[lo : lo + chunk].astype(np.int64)
        rows = np.repeat(ef, m, axis=1)
        cols = np.tile(ef, (1, m))
        keys = (rows * n_dof + cols).ravel()
        found = np.searchsorted(keys_unique, keys).clip(max=max(nnz - 1, 0))
        ok = (rows.ravel() >= 0) & (cols.ravel() >= 0)
        p = np.where(ok, found, nnz).astype(np.int32)
        pos[:, lo : lo + ef.shape[0]] = p.reshape(ef.shape[0], m * m).T
    return indptr, indices, pos


def _assemble_csr_data(scale_active: Any, ke_flat: Any, pos: Any, nnz: int) -> _F64:
    """Fill CSR data with one scatter pass per local-stiffness entry pair.

    Coerces inputs to host numpy so it stays correct (if slow) as the
    fallback for device-array backends.
    """
    sa = np.asarray(scale_active, dtype=np.float64)
    ke = np.asarray(ke_flat, dtype=np.float64)
    p = np.asarray(pos)
    data = np.zeros(nnz + 1)
    for k in range(p.shape[0]):
        np.add.at(data, p[k], ke[k] * sa)
    return data[:-1]


register_kernel("assemble_csr_data", "generic", _assemble_csr_data)


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
    ) -> None:
        self.mesh = mesh
        self.material = material
        self.mode = mode
        dim = mesh.dim
        if dim == 3 and mode != "plane_stress":
            raise FemError("mode is a 2D setting; remove it for 3D meshes")
        if mesh.element_kind not in ("quad4", "hex8"):
            raise FemError(f"unsupported element kind {mesh.element_kind!r}")

        self._d = _d_matrix(material, dim, mode)
        self._ke = _element_stiffness(self._d, mesh.spacing)
        self._ke.flags.writeable = False
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
        self._ke_flat = self._ke.ravel()
        self._indptr, self._indices, self._pos = _csr_pattern_positions(
            self._edofs_free.astype(np.int64), self._n_dof
        )
        self._nnz = int(self._indices.size)

        self._loads = self._build_loads(loads, cn, dim, n_cdof, free)
        self._loads.flags.writeable = False

    @property
    def element_stiffness(self) -> _F64:
        """The shared element stiffness matrix (read-only)."""
        return self._ke

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
        flat = [isinstance(item, PointLoad | SurfaceTraction | BodyForce) for item in loads]
        if all(flat):
            cases: list[Sequence[Load]] = [loads]  # type: ignore[list-item]
        elif any(flat):
            raise FemError("loads must be either one case of Loads or a list of cases")
        else:
            cases = list(loads)  # type: ignore[arg-type]
            for j, case in enumerate(cases):
                if not case:
                    raise FemError(f"load case {j} is empty")
                if not all(isinstance(it, PointLoad | SurfaceTraction | BodyForce) for it in case):
                    raise FemError(f"load case {j} contains non-Load entries")
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
        if bool((arr < 0.0).any()):
            raise FemError("scale contains negative values")
        return arr

    def assemble(self, scale: Any) -> SparseMatrix:
        """Free-DOF stiffness matrix for per-element ``scale`` multipliers."""
        arr = self._check_scale(scale)
        sa = arr[self.mesh.active_elements]
        backend = active_backend()
        kernel = get_kernel("assemble_csr_data")
        data = kernel(backend.asarray(sa), self._ke_flat, self._pos, self._nnz)
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
