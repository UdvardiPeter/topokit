# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""The 2D reference problems (MBB beam and cantilever), 88-line setup."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from topokit.fem import LinearElasticity, Material, PointLoad, UnstructuredElasticity
from topokit.mesh import StructuredGrid, UnstructuredMesh
from topokit.optimizers import MMA, OC, Optimizer
from topokit.parametrization import SIMP, RadialDensityFilter
from topokit.problem import Problem
from topokit.responses import Compliance, Volume
from topokit.selection import Box, NearPoint, PlaneSlab

Builder = Callable[..., Problem]


def _grid(nelx: int, nely: int) -> StructuredGrid:
    return StructuredGrid.box(size=(float(nelx), float(nely)), shape=(nelx, nely))


def _tri_mesh(nelx: int, nely: int, size: tuple[float, float]) -> UnstructuredMesh:
    """Quad-split triangulation on an ``(nelx+1) x (nely+1)`` x-fastest lattice.

    Each quad is split into two tri3s along the same diagonal (the
    ``[0,1,2],[0,2,3]`` winding). One fixed diagonal makes the mesh
    directionally biased and not mirror-symmetric about the box centrelines:
    no element has a mirror twin, so a symmetric problem on this mesh has an
    asymmetric optimum. Deliberate -- a frozen reference only has to be
    deterministic -- but it is part of why the tri3 and quad4 cantilever
    compliances differ (see ``test_tri3_vs_quad4_cantilever_agree``).
    """
    sx, sy = size
    nx1 = nelx + 1
    xx, yy = np.meshgrid(np.linspace(0.0, sx, nx1), np.linspace(0.0, sy, nely + 1), indexing="ij")
    nodes = np.column_stack([xx.ravel(order="F"), yy.ravel(order="F")])

    ii, jj = np.meshgrid(np.arange(nelx), np.arange(nely), indexing="ij")
    corner = ii.ravel(order="F") + nx1 * jj.ravel(order="F")
    quads = np.stack([corner, corner + 1, corner + 1 + nx1, corner + nx1], axis=1)
    elements = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]], axis=0)
    return UnstructuredMesh(nodes, elements.astype(np.int64), "tri3")


# The six Kuhn (Freudenthal) tets a cube splits into: each is one monotone
# corner-to-corner path from (0,0,0) to (1,1,1), given as (dx,dy,dz) corner
# offsets, ordered so the signed volume is positive (derived once, verified
# by `UnstructuredMesh`'s construction validation).
_KUHN_TETS: tuple[tuple[tuple[int, int, int], ...], ...] = (
    ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)),
    ((0, 0, 0), (1, 0, 1), (1, 0, 0), (1, 1, 1)),
    ((0, 0, 0), (1, 1, 0), (0, 1, 0), (1, 1, 1)),
    ((0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1)),
    ((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1)),
    ((0, 0, 0), (0, 1, 1), (0, 0, 1), (1, 1, 1)),
)

# tet4 local edge order VTK_QUADRATIC_TETRA pins its midside columns to
# (`topokit.mesh`'s module docstring): (0,1),(1,2),(2,0),(0,3),(1,3),(2,3).
_TET_LOCAL_EDGES: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))


def _tet_mesh(
    nelx: int, nely: int, nelz: int, size: tuple[float, float, float]
) -> UnstructuredMesh:
    """6-tet Kuhn split of an x-fastest lattice over the box.

    Each of the ``nelx*nely*nelz`` cubes splits into 6 tet4s along the fixed
    diagonal (0,0,0)-(1,1,1); every tet's vertex order is pinned so the
    signed volume comes out positive.
    """
    sx, sy, sz = size
    nx1, ny1 = nelx + 1, nely + 1
    xx, yy, zz = np.meshgrid(
        np.linspace(0.0, sx, nx1),
        np.linspace(0.0, sy, ny1),
        np.linspace(0.0, sz, nelz + 1),
        indexing="ij",
    )
    nodes = np.column_stack([xx.ravel(order="F"), yy.ravel(order="F"), zz.ravel(order="F")])

    ii, jj, kk = np.meshgrid(np.arange(nelx), np.arange(nely), np.arange(nelz), indexing="ij")
    base = ii.ravel(order="F") + nx1 * jj.ravel(order="F") + nx1 * ny1 * kk.ravel(order="F")
    tets = [
        np.stack([base + dx + nx1 * dy + nx1 * ny1 * dz for dx, dy, dz in path], axis=1)
        for path in _KUHN_TETS
    ]
    elements = np.concatenate(tets, axis=0)
    return UnstructuredMesh(nodes, elements.astype(np.int64), "tet4")


def _tet10_mesh(
    nelx: int, nely: int, nelz: int, size: tuple[float, float, float]
) -> UnstructuredMesh:
    """`_tet_mesh` upgraded to tet10: one midpoint node per unique tet4 edge.

    Connectivity is VTK_QUADRATIC_TETRA order: 4 corners then the midsides of
    edges (0,1),(1,2),(2,0),(0,3),(1,3),(2,3).
    """
    tet4 = _tet_mesh(nelx, nely, nelz, size)
    corners = tet4.nodes
    elements = tet4.element_nodes

    local_edges: npt.NDArray[np.int64] = np.array(_TET_LOCAL_EDGES, dtype=np.int64)
    pairs = np.sort(elements[:, local_edges], axis=2)  # (n_elements, 6, 2), canonical per edge
    unique_edges, inverse = np.unique(pairs.reshape(-1, 2), axis=0, return_inverse=True)
    midsides = inverse.reshape(elements.shape[0], 6) + corners.shape[0]

    midpoints = (corners[unique_edges[:, 0]] + corners[unique_edges[:, 1]]) / 2.0
    nodes = np.concatenate([corners, midpoints], axis=0)
    tet10_elements = np.concatenate([elements, midsides.astype(np.int64)], axis=1)
    return UnstructuredMesh(nodes, tet10_elements, "tet10")


def mbb(
    nelx: int,
    nely: int,
    *,
    volfrac: float = 0.5,
    penal: float = 3.0,
    rmin: float = 2.4,
    optimizer: Optimizer,
) -> Problem:
    """Half-MBB beam: left x-rollers, bottom-right y-support, top-left load."""
    grid = _grid(nelx, nely)
    left = PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0), tol=1e-9)
    bottom_right = NearPoint((float(nelx), 0.0))
    top_left = NearPoint((0.0, float(nely)))
    model = LinearElasticity(
        grid,
        Material(E=1.0, nu=0.3, rho=1.0),
        supports=[(left, "x"), (bottom_right, "y")],
        loads=[PointLoad(top_left, (0.0, -1.0))],
    )
    chain = RadialDensityFilter(radius=rmin) | SIMP(p=penal)
    return Problem(
        model, chain, objective=Compliance(), constraints=[Volume() <= volfrac], optimizer=optimizer
    )


def cantilever(
    nelx: int,
    nely: int,
    *,
    volfrac: float = 0.4,
    penal: float = 3.0,
    rmin: float = 2.4,
    optimizer: Optimizer,
) -> Problem:
    """Cantilever: left edge fully fixed, downward load at the mid-right edge."""
    grid = _grid(nelx, nely)
    left = PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0), tol=1e-9)
    mid_right = NearPoint((float(nelx), nely / 2.0))
    model = LinearElasticity(
        grid,
        Material(E=1.0, nu=0.3, rho=1.0),
        supports=[(left, "all")],
        loads=[PointLoad(mid_right, (0.0, -1.0))],
    )
    chain = RadialDensityFilter(radius=rmin) | SIMP(p=penal)
    return Problem(
        model, chain, objective=Compliance(), constraints=[Volume() <= volfrac], optimizer=optimizer
    )


def cantilever_3d(
    nelx: int,
    nely: int,
    nelz: int,
    *,
    volfrac: float = 0.3,
    penal: float = 3.0,
    rmin: float = 1.5,
    optimizer: Optimizer,
) -> Problem:
    """3D cantilever: left face fully fixed, downward load at the tip face centre."""
    grid = StructuredGrid.box(
        size=(float(nelx), float(nely), float(nelz)), shape=(nelx, nely, nelz)
    )
    left = PlaneSlab(point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), tol=1e-9)
    tip = NearPoint((float(nelx), nely / 2.0, nelz / 2.0))
    model = LinearElasticity(
        grid,
        Material(E=1.0, nu=0.3, rho=1.0),
        supports=[(left, "all")],
        loads=[PointLoad(tip, (0.0, -1.0, 0.0))],
    )
    chain = RadialDensityFilter(radius=rmin) | SIMP(p=penal)
    return Problem(
        model, chain, objective=Compliance(), constraints=[Volume() <= volfrac], optimizer=optimizer
    )


def cantilever_tri(
    nelx: int,
    nely: int,
    *,
    volfrac: float = 0.4,
    penal: float = 3.0,
    rmin: float = 2.4,
    optimizer: Optimizer,
) -> Problem:
    """Cantilever on a tri3 mesh: same geometry/BCs as `cantilever`, split into triangles."""
    mesh = _tri_mesh(nelx, nely, (float(nelx), float(nely)))
    left = PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0), tol=1e-9)
    mid_right = NearPoint((float(nelx), nely / 2.0))
    model = UnstructuredElasticity(
        mesh,
        Material(E=1.0, nu=0.3, rho=1.0),
        supports=[(left, "all")],
        loads=[PointLoad(mid_right, (0.0, -1.0))],
    )
    chain = RadialDensityFilter(radius=rmin) | SIMP(p=penal)
    return Problem(
        model, chain, objective=Compliance(), constraints=[Volume() <= volfrac], optimizer=optimizer
    )


def cantilever_tet4(
    nelx: int,
    nely: int,
    nelz: int,
    *,
    volfrac: float = 0.3,
    penal: float = 3.0,
    rmin: float = 1.5,
    optimizer: Optimizer,
) -> Problem:
    """3D cantilever on a tet4 mesh: same geometry/BCs as `cantilever_3d`, Kuhn-split."""
    mesh = _tet_mesh(nelx, nely, nelz, (float(nelx), float(nely), float(nelz)))
    left = PlaneSlab(point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), tol=1e-9)
    tip = NearPoint((float(nelx), nely / 2.0, nelz / 2.0))
    model = UnstructuredElasticity(
        mesh,
        Material(E=1.0, nu=0.3, rho=1.0),
        supports=[(left, "all")],
        loads=[PointLoad(tip, (0.0, -1.0, 0.0))],
    )
    chain = RadialDensityFilter(radius=rmin) | SIMP(p=penal)
    return Problem(
        model, chain, objective=Compliance(), constraints=[Volume() <= volfrac], optimizer=optimizer
    )


def cantilever_tet10(
    nelx: int,
    nely: int,
    nelz: int,
    *,
    volfrac: float = 0.3,
    penal: float = 3.0,
    rmin: float = 1.5,
    optimizer: Optimizer,
) -> Problem:
    """3D cantilever on a tet10 mesh: same geometry/BCs as `cantilever_3d`, Kuhn-split."""
    mesh = _tet10_mesh(nelx, nely, nelz, (float(nelx), float(nely), float(nelz)))
    left = PlaneSlab(point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), tol=1e-9)
    tip = NearPoint((float(nelx), nely / 2.0, nelz / 2.0))
    model = UnstructuredElasticity(
        mesh,
        Material(E=1.0, nu=0.3, rho=1.0),
        supports=[(left, "all")],
        loads=[PointLoad(tip, (0.0, -1.0, 0.0))],
    )
    chain = RadialDensityFilter(radius=rmin) | SIMP(p=penal)
    return Problem(
        model, chain, objective=Compliance(), constraints=[Volume() <= volfrac], optimizer=optimizer
    )


def michell(
    nelx: int,
    nely: int,
    *,
    volfrac: float = 0.5,
    penal: float = 3.0,
    rmin: float = 2.4,
    optimizer: Optimizer,
) -> Problem:
    """Michell cantilever: a left-edge support band around mid-height, tip load.

    The support is a band (not a point) to remove the rigid-body rotation a single
    fixed node would leave; the mid-height support + mid-height load make the setup
    mirror-symmetric about the horizontal centreline, so the optimum is too.
    """
    grid = _grid(nelx, nely)
    band = Box((0.0, nely / 3.0), (0.0, 2.0 * nely / 3.0), tol=1e-9)
    mid_right = NearPoint((float(nelx), nely / 2.0))
    model = LinearElasticity(
        grid,
        Material(E=1.0, nu=0.3, rho=1.0),
        supports=[(band, "all")],
        loads=[PointLoad(mid_right, (0.0, -1.0))],
    )
    chain = RadialDensityFilter(radius=rmin) | SIMP(p=penal)
    return Problem(
        model, chain, objective=Compliance(), constraints=[Volume() <= volfrac], optimizer=optimizer
    )


# The benchmark matrix, the single source of truth shared by the regression
# suite and the reference-regeneration script.
BUILDERS: dict[str, Builder] = {
    "mbb": mbb,
    "cantilever": cantilever,
    "cantilever_tri": cantilever_tri,
}
CASES: list[tuple[str, int, int, str]] = [
    (name, nelx, nely, opt)
    for name in ("mbb", "cantilever")
    for nelx, nely in ((60, 20), (150, 50))
    for opt in ("oc", "mma")
] + [("cantilever_tri", 60, 20, opt) for opt in ("oc", "mma")]


def make_optimizer(name: str) -> Optimizer:
    """Map a case's optimizer name to a configured instance."""
    return OC(move=0.2) if name == "oc" else MMA()


@dataclass(frozen=True)
class FullCase:
    """A nightly (Tier-3 full) case: builder + integer size kwargs + optimizers."""

    key: str
    build: Builder
    kwargs: dict[str, int]
    optimizers: tuple[str, ...]


# The nightly full-suite matrix: 3D cantilever (OC only — OC/MMA agreement is
# already locked in 2D) and the 2D Michell (OC + MMA, cheap).
FULL_CASES: list[FullCase] = [
    FullCase(
        "cantilever_3d_24x12x12", cantilever_3d, {"nelx": 24, "nely": 12, "nelz": 12}, ("oc",)
    ),
    FullCase("michell_90x30", michell, {"nelx": 90, "nely": 30}, ("oc", "mma")),
    FullCase(
        "cantilever_tet4_16x8x8", cantilever_tet4, {"nelx": 16, "nely": 8, "nelz": 8}, ("oc",)
    ),
    # 10x5x5, not 12x6x6: measured 12x6x6 at 2.27x the hex cantilever_3d_24x12x12
    # case's wall time, over the ~2x nightly budget; 10x5x5 measured at 0.72x.
    FullCase(
        "cantilever_tet10_10x5x5", cantilever_tet10, {"nelx": 10, "nely": 5, "nelz": 5}, ("oc",)
    ),
]
