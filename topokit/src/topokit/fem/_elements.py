# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Per-element stiffness and strain-displacement kernels for simplex elements.

Supports ``tri3`` (2D, VTK_TRIANGLE), ``tet4`` (3D, VTK_TETRA), and ``tet10``
(3D, VTK_QUADRATIC_TETRA), following the connectivity conventions documented
in :mod:`topokit.mesh`.

``tri3``/``tet4`` have constant strain-displacement matrices (a single
integration point suffices, exactly); ``tet10`` is integrated with the
standard 4-point degree-2 tetrahedron rule, evaluating the isoparametric
Jacobian from the full 10-node map at each Gauss point so curved (non
straight-edge) elements are handled correctly.

Engineering-strain row order mirrors ``fem._b_matrix``/``fem._d_matrix``
exactly: ``(xx, yy, xy)`` in 2D, ``(xx, yy, zz, xy, yz, zx)`` in 3D. DOFs are
node-major (``[u0, v0, (w0), u1, v1, (w1), ...]``). Getting the row order
wrong produces a stiffness matrix that still passes energy-only checks for
isotropic materials (D's normal-strain and shear blocks are individually
permutation-symmetric) but yields wrong per-component stress -- callers
needing stress must use this exact convention.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from topokit.fem import FemError

_F64 = npt.NDArray[np.float64]
_I64 = npt.NDArray[np.int64]

NDOF: dict[str, int] = {"tri3": 6, "tet4": 12, "tet10": 30}
NODES_PER_FACE: dict[str, int] = {"tri3": 2, "tet4": 3, "tet10": 6}

# Consistent nodal weights for constant traction on a straight boundary face
# (shape-function integration); multiply by traction * face area. tet10's
# corner weight is exactly 0: for a straight-sided quadratic triangle face,
# integrating a corner shape function N_i = L_i(2L_i - 1) against a constant
# gives 2*(area/6) - area/3 = 0 (area-coordinate moment formula).
_FACE_WEIGHTS: dict[str, tuple[float, ...]] = {
    "tri3": (0.5, 0.5),
    "tet4": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    "tet10": (0.0, 0.0, 0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
}

# Reference-tet4 shape-function gradients d N_i/d(r, s, t), node order
# matching N1 = 1-r-s-t, N2 = r, N3 = s, N4 = t.
_TET4_DN_REF = np.array([[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

# tet10 midside-node (index 4..9) corner-index pairs, VTK_QUADRATIC_TETRA
# order: edges (0,1), (1,2), (2,0), (0,3), (1,3), (2,3).
_TET10_MIDSIDE_PAIRS = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))

# Standard 4-point degree-2 tetrahedron quadrature (e.g. Zienkiewicz &
# Taylor): barycentric points (a, b, b, b) and permutations, natural
# (dr ds dt) weight 1/24 each -- i.e. reference-tet-volume (1/6) / 4 points,
# summing to the reference volume as required.
_TET10_GP_A = 0.5854101966249685
_TET10_GP_B = 0.1381966011250105
_TET10_GAUSS_RST: tuple[tuple[float, float, float], ...] = (
    (_TET10_GP_B, _TET10_GP_B, _TET10_GP_B),
    (_TET10_GP_A, _TET10_GP_B, _TET10_GP_B),
    (_TET10_GP_B, _TET10_GP_A, _TET10_GP_B),
    (_TET10_GP_B, _TET10_GP_B, _TET10_GP_A),
)
_TET10_GAUSS_WEIGHT = 1.0 / 24.0


def face_weights(kind: str) -> _F64:
    """Consistent nodal weights for constant traction on a boundary face.

    Multiply by ``traction * area`` to distribute a constant traction to a
    face's nodes exactly (consistent shape-function integration on a
    straight face). Sums to 1 for every kind.
    """
    try:
        weights = _FACE_WEIGHTS[kind]
    except KeyError:
        raise FemError(f"unknown element kind {kind!r}") from None
    return np.array(weights, dtype=np.float64)


def _apply_d(b: _F64, d: _F64) -> _F64:
    """``B^T D B`` for a stack of B matrices, shape ``(n_el, ndof, ndof)``."""
    return np.asarray(np.einsum("eri,rs,esj->eij", b, d, b))


def _b_stack(grad: _F64, dim: int) -> _F64:
    """Strain-displacement stack from physical shape-function gradients.

    ``grad`` has shape ``(n_el, nen, dim)``; DOFs are node-major. Row order
    mirrors ``fem._b_matrix``: ``(xx, yy, xy)`` in 2D, ``(xx, yy, zz, xy, yz,
    zx)`` in 3D.
    """
    n_el, nen, _ = grad.shape
    if dim == 2:
        b = np.zeros((n_el, 3, 2 * nen))
        b[:, 0, 0::2] = grad[:, :, 0]
        b[:, 1, 1::2] = grad[:, :, 1]
        b[:, 2, 0::2] = grad[:, :, 1]
        b[:, 2, 1::2] = grad[:, :, 0]
        return b
    b = np.zeros((n_el, 6, 3 * nen))
    b[:, 0, 0::3] = grad[:, :, 0]
    b[:, 1, 1::3] = grad[:, :, 1]
    b[:, 2, 2::3] = grad[:, :, 2]
    b[:, 3, 0::3] = grad[:, :, 1]
    b[:, 3, 1::3] = grad[:, :, 0]
    b[:, 4, 1::3] = grad[:, :, 2]
    b[:, 4, 2::3] = grad[:, :, 1]
    b[:, 5, 0::3] = grad[:, :, 2]
    b[:, 5, 2::3] = grad[:, :, 0]
    return b


def _tri3_measure_and_grad(nodes: _F64, elements: _I64) -> tuple[_F64, _F64]:
    """Signed area and constant shape-function gradients, per element.

    ``A = 0.5*det([[x2-x1, x3-x1], [y2-y1, y3-y1]])``; ``dN_i = (b_i, c_i) /
    (2A)`` with ``b_1 = y2-y3, c_1 = x3-x2`` (cyclic).
    """
    p = nodes[elements]  # (n_el, 3, 2)
    x1, y1 = p[:, 0, 0], p[:, 0, 1]
    x2, y2 = p[:, 1, 0], p[:, 1, 1]
    x3, y3 = p[:, 2, 0], p[:, 2, 1]
    area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    b = np.stack([y2 - y3, y3 - y1, y1 - y2], axis=-1)
    c = np.stack([x3 - x2, x1 - x3, x2 - x1], axis=-1)
    grad = np.stack([b, c], axis=-1) / (2.0 * area)[:, None, None]
    return area, grad


def _tet4_measure_and_grad(nodes: _F64, elements: _I64) -> tuple[_F64, _F64]:
    """Signed volume and constant shape-function gradients, per element.

    ``V = det(J)/6`` with ``J = [x2-x1; x3-x1; x4-x1]``; gradients come from
    ``grad N = J^-1 . dN_ref``.
    """
    p = nodes[elements]  # (n_el, 4, 3)
    e1 = p[:, 1] - p[:, 0]
    e2 = p[:, 2] - p[:, 0]
    e3 = p[:, 3] - p[:, 0]
    j = np.stack([e1, e2, e3], axis=1)  # (n_el, 3, 3), rows e1, e2, e3
    volume = np.linalg.det(j) / 6.0
    j_inv = np.linalg.inv(j)
    grad = np.einsum("eca,ia->eic", j_inv, _TET4_DN_REF)
    return volume, grad


def _tet10_dn_ref(r: float, s: float, t: float) -> _F64:
    """Reference-coordinate shape-function gradients at one (r, s, t) point.

    Barycentric ``L = (1-r-s-t, r, s, t)``; vertex ``N_i = L_i(2L_i-1)``,
    midside ``N_ij = 4 L_i L_j``. Node order: 4 vertices then midsides on
    edges (0,1),(1,2),(2,0),(0,3),(1,3),(2,3) (VTK_QUADRATIC_TETRA).
    """
    ell = np.array([1.0 - r - s - t, r, s, t])
    d_ell = np.array([[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    dn = np.zeros((10, 3))
    dn[:4] = (4.0 * ell[:, None] - 1.0) * d_ell
    for m, (i, j) in enumerate(_TET10_MIDSIDE_PAIRS):
        dn[4 + m] = 4.0 * (d_ell[i] * ell[j] + ell[i] * d_ell[j])
    return dn


def _tet10_grad_and_detj(nodes: _F64, elements: _I64, dn_ref: _F64) -> tuple[_F64, _F64]:
    """Physical shape-function gradients and ``det J``, at one reference point.

    The Jacobian is built from the full 10-node isoparametric map
    (``J[e] = dn_ref^T @ nodes_of(e)``), not the corner tetrahedron, so
    curved (non-straight-edge) elements are handled correctly.
    """
    x = nodes[elements]  # (n_el, 10, 3)
    j = np.einsum("ia,eib->eab", dn_ref, x)  # (n_el, 3, 3)
    det_j = np.linalg.det(j)
    j_inv = np.linalg.inv(j)
    grad = np.einsum("eca,ia->eic", j_inv, dn_ref)
    return grad, det_j


def stiffness_stack(kind: str, d: _F64, nodes: _F64, elements: _I64) -> _F64:
    """Per-element stiffness matrices, shape ``(n_elements, ndof, ndof)``.

    ``d`` is the material matrix from ``fem._d_matrix`` (3x3 in 2D, 6x6 in
    3D); ``nodes`` is ``(n_nodes, dim)``, ``elements`` the connectivity
    ``(n_elements, nodes_per_element)``.
    """
    if kind == "tri3":
        area, grad = _tri3_measure_and_grad(nodes, elements)
        b = _b_stack(grad, dim=2)
        return area[:, None, None] * _apply_d(b, d)
    if kind == "tet4":
        volume, grad = _tet4_measure_and_grad(nodes, elements)
        b = _b_stack(grad, dim=3)
        return volume[:, None, None] * _apply_d(b, d)
    if kind == "tet10":
        n_el = elements.shape[0]
        ke = np.zeros((n_el, 30, 30))
        # 4 Gauss points: a small fixed-size Python loop reads directly as
        # the quadrature sum, instead of hiding it behind a vectorized
        # point axis.
        for r, s, t in _TET10_GAUSS_RST:
            dn_ref = _tet10_dn_ref(r, s, t)
            grad, det_j = _tet10_grad_and_detj(nodes, elements, dn_ref)
            b = _b_stack(grad, dim=3)
            ke += _TET10_GAUSS_WEIGHT * np.abs(det_j)[:, None, None] * _apply_d(b, d)
        return ke
    raise FemError(f"unknown element kind {kind!r}")


def centroid_b_stack(kind: str, nodes: _F64, elements: _I64) -> _F64:
    """Per-element strain-displacement matrix at the element centroid.

    For stress display: ``stress = D @ (B_centroid @ u_e)``. ``tri3``/
    ``tet4`` have a single constant ``B``; ``tet10`` evaluates ``B`` at the
    barycentric centroid ``(1/4, 1/4, 1/4, 1/4)``.
    """
    if kind == "tri3":
        _, grad = _tri3_measure_and_grad(nodes, elements)
        return _b_stack(grad, dim=2)
    if kind == "tet4":
        _, grad = _tet4_measure_and_grad(nodes, elements)
        return _b_stack(grad, dim=3)
    if kind == "tet10":
        dn_ref = _tet10_dn_ref(0.25, 0.25, 0.25)
        grad, _ = _tet10_grad_and_detj(nodes, elements, dn_ref)
        return _b_stack(grad, dim=3)
    raise FemError(f"unknown element kind {kind!r}")
