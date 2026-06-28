# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""The 2D reference problems (MBB beam and cantilever), 88-line setup."""

from __future__ import annotations

from collections.abc import Callable

from topokit.fem import LinearElasticity, Material, PointLoad
from topokit.mesh import StructuredGrid
from topokit.optimizers import MMA, OC, Optimizer
from topokit.parametrization import SIMP, RadialDensityFilter
from topokit.problem import Problem
from topokit.responses import Compliance, Volume
from topokit.selection import NearPoint, PlaneSlab

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
