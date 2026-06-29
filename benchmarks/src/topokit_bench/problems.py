# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""The 2D reference problems (MBB beam and cantilever), 88-line setup."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from topokit.fem import LinearElasticity, Material, PointLoad
from topokit.mesh import StructuredGrid
from topokit.optimizers import MMA, OC, Optimizer
from topokit.parametrization import SIMP, RadialDensityFilter
from topokit.problem import Problem
from topokit.responses import Compliance, Volume
from topokit.selection import Box, NearPoint, PlaneSlab

Builder = Callable[..., Problem]


def _grid(nelx: int, nely: int) -> StructuredGrid:
    return StructuredGrid.box(size=(float(nelx), float(nely)), shape=(nelx, nely))


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
BUILDERS: dict[str, Builder] = {"mbb": mbb, "cantilever": cantilever}
CASES: list[tuple[str, int, int, str]] = [
    (name, nelx, nely, opt)
    for name in BUILDERS
    for nelx, nely in ((60, 20), (150, 50))
    for opt in ("oc", "mma")
]


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
]
