# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Tests for the simplex element library: stiffness, B, and face weights.

The element-level arbiter for every constant in ``fem._elements`` is the
strain-energy patch test: a single element under a displacement field its
shape functions can represent exactly must reproduce the analytic strain
energy. Row-order tests pin the engineering-strain convention directly
(``B @ u`` against known distinct strain components) because an isotropic
material's block-symmetric ``D`` cannot distinguish a row permutation
through energy alone -- a wrong xx/yy/zz or xy/yz/zx assignment still
integrates to the same (wrong-looking-right) energy.
"""

import numpy as np
import pytest

from topokit.fem import FemError, Material, _d_matrix
from topokit.fem._elements import (
    NDOF,
    NODES_PER_FACE,
    body_weights,
    centroid_b_stack,
    face_weights,
    stiffness_stack,
)

_MATERIAL = Material(E=1.0, nu=0.3, rho=1.0)

_TRI3_NODES = np.array([[0.0, 0.0], [3.0, 0.0], [1.0, 2.0]])
_TRI3_ELEMENTS = np.array([[0, 1, 2]])

_TET4_NODES = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
_TET4_ELEMENTS = np.array([[0, 1, 2, 3]])

# Corner tet10: straight edges, midsides at exact edge midpoints, node order
# VTK_QUADRATIC_TETRA (4 vertices, then midsides on edges (0,1),(1,2),(2,0),
# (0,3),(1,3),(2,3)). Physical coordinates equal the reference (r, s, t) here,
# which keeps the analytic energy derivations below tractable.
_TET10_NODES = np.array(
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
_TET10_ELEMENTS = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])

_GEOMETRY: dict[str, tuple[np.ndarray, np.ndarray, int]] = {
    "tri3": (_TRI3_NODES, _TRI3_ELEMENTS, 2),
    "tet4": (_TET4_NODES, _TET4_ELEMENTS, 3),
    "tet10": (_TET10_NODES, _TET10_ELEMENTS, 3),
}

_KINDS = ("tri3", "tet4", "tet10")


def _d(dim: int) -> np.ndarray:
    # 3D branch of _d_matrix ignores mode; plane_stress is only meaningful for dim=2.
    return np.asarray(_d_matrix(_MATERIAL, dim, "plane_stress"))


def test_ndof_and_nodes_per_face_tables() -> None:
    assert NDOF == {"tri3": 6, "tet4": 12, "tet10": 30}
    assert NODES_PER_FACE == {"tri3": 2, "tet4": 3, "tet10": 6}


@pytest.mark.parametrize("kind", _KINDS)
def test_stiffness_stack_shape(kind: str) -> None:
    nodes, elements, dim = _GEOMETRY[kind]
    ke = stiffness_stack(kind, _d(dim), nodes, elements)
    n = NDOF[kind]
    assert ke.shape == (1, n, n)


@pytest.mark.parametrize("kind", _KINDS)
def test_stiffness_is_symmetric(kind: str) -> None:
    nodes, elements, dim = _GEOMETRY[kind]
    ke = stiffness_stack(kind, _d(dim), nodes, elements)[0]
    np.testing.assert_allclose(ke, ke.T, atol=1e-12)


@pytest.mark.parametrize("kind", _KINDS)
def test_rigid_body_translation_nullity(kind: str) -> None:
    nodes, elements, dim = _GEOMETRY[kind]
    ke = stiffness_stack(kind, _d(dim), nodes, elements)[0]
    nen = elements.shape[1]
    for comp in range(dim):
        mode = np.zeros(nen * dim)
        mode[comp::dim] = 1.0
        np.testing.assert_allclose(ke @ mode, 0.0, atol=1e-9)


def _rotation_modes(nodes: np.ndarray, elements: np.ndarray, dim: int) -> list[np.ndarray]:
    """Node-major rigid-rotation DOF vectors, ``u = omega x coords`` per axis."""
    coords = nodes[elements[0]]
    if dim == 2:
        mode = np.zeros_like(coords)
        mode[:, 0] = -coords[:, 1]
        mode[:, 1] = coords[:, 0]
        return [mode.ravel()]
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    rot_x = np.zeros_like(coords)
    rot_x[:, 1] = -z
    rot_x[:, 2] = y
    rot_y = np.zeros_like(coords)
    rot_y[:, 0] = z
    rot_y[:, 2] = -x
    rot_z = np.zeros_like(coords)
    rot_z[:, 0] = -y
    rot_z[:, 1] = x
    return [rot_x.ravel(), rot_y.ravel(), rot_z.ravel()]


@pytest.mark.parametrize("kind", _KINDS)
def test_rigid_body_rotation_nullity(kind: str) -> None:
    nodes, elements, dim = _GEOMETRY[kind]
    ke = stiffness_stack(kind, _d(dim), nodes, elements)[0]
    for mode in _rotation_modes(nodes, elements, dim):
        np.testing.assert_allclose(ke @ mode, 0.0, atol=1e-9)


@pytest.mark.parametrize("kind", _KINDS)
def test_stiffness_is_positive_semidefinite(kind: str) -> None:
    nodes, elements, dim = _GEOMETRY[kind]
    ke = stiffness_stack(kind, _d(dim), nodes, elements)[0]
    eig = np.linalg.eigvalsh(ke)
    scaled = eig / np.max(np.abs(eig))
    assert scaled.min() >= -1e-9


def test_tri3_strain_row_order_matches_d_matrix() -> None:
    """B @ u for a linear field pins the (xx, yy, xy) row order directly.

    An energy-only check cannot: isotropic D is invariant under permuting
    the two normal-strain rows or relabeling the single shear row alone.
    """
    nodes, elements, _dim = _GEOMETRY["tri3"]
    exx, eyy, gxy = 1.0, 2.0, 3.0
    x, y = nodes[:, 0], nodes[:, 1]
    ux = exx * x + 0.5 * gxy * y
    uy = eyy * y + 0.5 * gxy * x
    u = np.stack([ux, uy], axis=-1).ravel()
    b = centroid_b_stack("tri3", nodes, elements)[0]
    np.testing.assert_allclose(b @ u, [exx, eyy, gxy], atol=1e-9)


def test_tet4_strain_row_order_matches_d_matrix() -> None:
    """B @ u for a linear field pins the (xx, yy, zz, xy, yz, zx) row order."""
    nodes, elements, _dim = _GEOMETRY["tet4"]
    exx, eyy, ezz, gxy, gyz, gzx = 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
    x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    ux = exx * x + 0.5 * gxy * y + 0.5 * gzx * z
    uy = eyy * y + 0.5 * gxy * x + 0.5 * gyz * z
    uz = ezz * z + 0.5 * gyz * y + 0.5 * gzx * x
    u = np.stack([ux, uy, uz], axis=-1).ravel()
    b = centroid_b_stack("tet4", nodes, elements)[0]
    np.testing.assert_allclose(b @ u, [exx, eyy, ezz, gxy, gyz, gzx], atol=1e-9)


def test_tet10_strain_row_order_matches_d_matrix() -> None:
    """Same row-order pin as tet4; a linear field is exact for tet10 too."""
    nodes, elements, _dim = _GEOMETRY["tet10"]
    exx, eyy, ezz, gxy, gyz, gzx = 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
    x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    ux = exx * x + 0.5 * gxy * y + 0.5 * gzx * z
    uy = eyy * y + 0.5 * gxy * x + 0.5 * gyz * z
    uz = ezz * z + 0.5 * gyz * y + 0.5 * gzx * x
    u = np.stack([ux, uy, uz], axis=-1).ravel()
    b = centroid_b_stack("tet10", nodes, elements)[0]
    np.testing.assert_allclose(b @ u, [exx, eyy, ezz, gxy, gyz, gzx], atol=1e-9)


def test_tri3_linear_patch_energy() -> None:
    nodes, elements, dim = _GEOMETRY["tri3"]
    d = _d(dim)
    exx, eyy, gxy = 0.7, -0.4, 0.5
    x, y = nodes[:, 0], nodes[:, 1]
    ux = exx * x + 0.5 * gxy * y
    uy = eyy * y + 0.5 * gxy * x
    u = np.stack([ux, uy], axis=-1).ravel()
    ke = stiffness_stack("tri3", d, nodes, elements)[0]
    energy = 0.5 * u @ ke @ u
    eps = np.array([exx, eyy, gxy])
    e1, e2 = nodes[1] - nodes[0], nodes[2] - nodes[0]
    area = 0.5 * abs(e1[0] * e2[1] - e1[1] * e2[0])
    analytic = 0.5 * eps @ d @ eps * area
    assert energy == pytest.approx(analytic, abs=1e-9)


def test_tet4_linear_patch_energy() -> None:
    nodes, elements, dim = _GEOMETRY["tet4"]
    d = _d(dim)
    exx, eyy, ezz, gxy, gyz, gzx = 0.7, -0.3, 0.2, 0.5, -0.4, 0.1
    x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    ux = exx * x + 0.5 * gxy * y + 0.5 * gzx * z
    uy = eyy * y + 0.5 * gxy * x + 0.5 * gyz * z
    uz = ezz * z + 0.5 * gyz * y + 0.5 * gzx * x
    u = np.stack([ux, uy, uz], axis=-1).ravel()
    ke = stiffness_stack("tet4", d, nodes, elements)[0]
    energy = 0.5 * u @ ke @ u
    eps = np.array([exx, eyy, ezz, gxy, gyz, gzx])
    e1, e2, e3 = nodes[1] - nodes[0], nodes[2] - nodes[0], nodes[3] - nodes[0]
    volume = abs(np.dot(e1, np.cross(e2, e3))) / 6.0
    analytic = 0.5 * eps @ d @ eps * volume
    assert energy == pytest.approx(analytic, abs=1e-9)


def test_tet10_quadratic_patch_energy() -> None:
    """A pure-quadratic field tet10 represents exactly; bending-representative.

    ``u = (x^2 + xy, y^2 + yz, z^2 + zx)`` gives linear strains
    ``(exx, eyy, ezz) = (2x+y, 2y+z, 2z+x)`` and ``(gxy, gyz, gzx) = (x, y, z)``.
    Integrating ``0.5 eps^T D eps`` over the unit corner tet (volume 1/6, using
    the moment formula for barycentric monomials) gives the closed form
    ``(9/20) lambda + (3/8) mu`` below; cross-checked against a 2e7-sample
    Monte-Carlo integration during derivation (agreement to ~1e-4).
    """
    nodes, elements, dim = _GEOMETRY["tet10"]
    d = _d(dim)
    x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    ux = x**2 + x * y
    uy = y**2 + y * z
    uz = z**2 + z * x
    u = np.stack([ux, uy, uz], axis=-1).ravel()
    ke = stiffness_stack("tet10", d, nodes, elements)[0]
    energy = 0.5 * u @ ke @ u
    e, nu = _MATERIAL.E, _MATERIAL.nu
    lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = e / (2.0 * (1.0 + nu))
    analytic = (9.0 / 20.0) * lam + (3.0 / 8.0) * mu
    assert energy == pytest.approx(analytic, abs=1e-9)


def test_tet10_inverted_element_raises() -> None:
    """A deliberately inverted tet10 raises instead of silently np.abs()-ing.

    Swapping vertices 0 and 1 (and the two midsides that touch them, so
    every midside stays at its true edge midpoint) is still a valid
    straight-sided tet10 -- just an odd permutation of the corner order, so
    its (constant, for a straight-sided element) Jacobian is negative at
    every Gauss point. The mesh constructor already rejects a non-positive
    signed volume outright, so this calls ``stiffness_stack`` directly to
    exercise the per-Gauss-point check inside it.
    """
    nodes, _elements, dim = _GEOMETRY["tet10"]
    inverted = np.array([[1, 0, 2, 3, 4, 6, 5, 8, 7, 9]])
    with pytest.raises(FemError, match="element ids"):
        stiffness_stack("tet10", _d(dim), nodes, inverted)


@pytest.mark.parametrize("kind", _KINDS)
def test_face_weights_sum_to_one(kind: str) -> None:
    w = face_weights(kind)
    assert w.shape == (NODES_PER_FACE[kind],)
    assert w.sum() == pytest.approx(1.0)


def test_tet10_face_corner_weights_are_zero() -> None:
    w = face_weights("tet10")
    np.testing.assert_array_equal(w[:3], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(w[3:], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])


def test_face_weights_unknown_kind_raises() -> None:
    with pytest.raises(FemError, match="kind"):
        face_weights("quad4")


@pytest.mark.parametrize("kind", _KINDS)
def test_body_weights_sum_to_one(kind: str) -> None:
    _nodes, elements, _dim = _GEOMETRY[kind]
    w = body_weights(kind)
    assert w.shape == (elements.shape[1],)
    assert w.sum() == pytest.approx(1.0)


def test_tet10_body_weights_negative_at_vertices_positive_at_midsides() -> None:
    """Derivation: barycentric moment formula int L1^a..L4^d dV = 6V a!b!c!d!/(a+b+c+d+3)!.

    Vertex ``N_i = L_i(2L_i-1)``: ``2*int(L_i^2) - int(L_i) = 2*(V/10) - V/4
    = -V/20``. Midside ``N_ij = 4 L_i L_j``: ``4*int(L_i L_j) = 4*(V/20) =
    V/5``. Cross-checked against ``test_body_weights_sum_to_one``: 4*(-1/20)
    + 6*(1/5) = 1.
    """
    w = body_weights("tet10")
    np.testing.assert_allclose(w[:4], [-1.0 / 20.0] * 4)
    np.testing.assert_allclose(w[4:], [1.0 / 5.0] * 6)


def test_body_weights_unknown_kind_raises() -> None:
    with pytest.raises(FemError, match="kind"):
        body_weights("quad4")


def test_stiffness_stack_unknown_kind_raises() -> None:
    with pytest.raises(FemError, match="kind"):
        stiffness_stack("quad4", np.eye(3), _TRI3_NODES, _TRI3_ELEMENTS)


def test_centroid_b_stack_unknown_kind_raises() -> None:
    with pytest.raises(FemError, match="kind"):
        centroid_b_stack("quad4", _TRI3_NODES, _TRI3_ELEMENTS)
