# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Regenerate the docs gallery images.

Run locally with `uv run --group gallery python scripts/gallery.py`; images are
committed so CI never renders them.

Each case function below builds its problem the same way the gallery page
shows it: geometry copied from ``topokit_bench.problems`` (the benchmark
suite's reference cases) translated into plain ``topokit`` calls, so a reader
can run the page's script with only ``topokit`` installed. The thermal case
reuses the ``HeatConduction``/``RampConductivity`` classes developed in
docs/content/extend/custom-physics.md, verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from pyvista import Plotter

from topokit import (
    MMA,
    OC,
    SIMP,
    BoundLink,
    Box,
    Compliance,
    DensityFilter,
    FieldSpec,
    LinearElasticity,
    LinkSpec,
    Material,
    Mesh,
    NearPoint,
    PlaneSlab,
    PointLoad,
    Problem,
    RadialDensityFilter,
    Result,
    Schedule,
    StructuredGrid,
    Study,
    Volume,
)
from topokit.backend import SparseMatrix, active_backend

_F64 = npt.NDArray[np.float64]

ROOT = Path(__file__).resolve().parents[2]
IMG = ROOT / "docs" / "content" / "gallery" / "img"
ASSETS = ROOT / "docs" / "content" / "assets"
IMG.mkdir(parents=True, exist_ok=True)
ASSETS.mkdir(parents=True, exist_ok=True)


def _report(label: str, out: Path, result: Result) -> None:
    print(
        f"{label}: compliance {result.objective:.2f} volume "
        f"{result.history['volume'][-1]:.3f} iterations {result.iterations} "
        f"wall {result.timing:.1f}s -> {out}"
    )


def mbb() -> Path:
    """Half-MBB beam, 150x50: left x-rollers, bottom-right y-support, top-left load."""
    nelx, nely = 150, 50
    mesh = StructuredGrid.box(size=(float(nelx), float(nely)), shape=(nelx, nely))
    left = PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0), tol=1e-9)
    bottom_right = NearPoint((float(nelx), 0.0))
    top_left = NearPoint((0.0, float(nely)))
    model = LinearElasticity(
        mesh,
        Material(E=1.0, nu=0.3, rho=1.0),
        supports=[(left, "x"), (bottom_right, "y")],
        loads=[PointLoad(top_left, force=(0.0, -1.0))],
    )
    chain = RadialDensityFilter(radius=2.4) | SIMP(p=3.0)
    problem = Problem(
        model,
        chain,
        objective=Compliance(),
        constraints=[Volume() <= 0.5],
        optimizer=OC(move=0.2),
    )
    result = Study(problem, schedule=Schedule.default(max_iter=60, tol=1e-3)).run()
    out = IMG / "mbb.png"
    fig = cast("Figure", result.view())
    fig.savefig(out, dpi=150, bbox_inches="tight")
    _report("mbb", out, result)
    return out


def cantilever() -> Path:
    """Cantilever, 120x40: left edge fully fixed, downward load at the mid-right edge."""
    nelx, nely = 120, 40
    mesh = StructuredGrid.box(size=(float(nelx), float(nely)), shape=(nelx, nely))
    left = PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0), tol=1e-9)
    mid_right = NearPoint((float(nelx), nely / 2.0))
    model = LinearElasticity(
        mesh,
        Material(E=1.0, nu=0.3, rho=1.0),
        supports=[(left, "all")],
        loads=[PointLoad(mid_right, force=(0.0, -1.0))],
    )
    chain = RadialDensityFilter(radius=2.4) | SIMP(p=3.0)
    problem = Problem(
        model,
        chain,
        objective=Compliance(),
        constraints=[Volume() <= 0.4],
        optimizer=OC(move=0.2),
    )
    result = Study(problem, schedule=Schedule.default(max_iter=60, tol=1e-3)).run()
    out = IMG / "cantilever.png"
    fig = cast("Figure", result.view())
    fig.savefig(out, dpi=150, bbox_inches="tight")
    _report("cantilever", out, result)
    return out


def michell() -> Path:
    """Michell cantilever, 90x30: a left-edge support band around mid-height, tip load."""
    nelx, nely = 90, 30
    mesh = StructuredGrid.box(size=(float(nelx), float(nely)), shape=(nelx, nely))
    band = Box((0.0, nely / 3.0), (0.0, 2.0 * nely / 3.0), tol=1e-9)
    mid_right = NearPoint((float(nelx), nely / 2.0))
    model = LinearElasticity(
        mesh,
        Material(E=1.0, nu=0.3, rho=1.0),
        supports=[(band, "all")],
        loads=[PointLoad(mid_right, force=(0.0, -1.0))],
    )
    chain = RadialDensityFilter(radius=2.4) | SIMP(p=3.0)
    problem = Problem(
        model,
        chain,
        objective=Compliance(),
        constraints=[Volume() <= 0.5],
        optimizer=MMA(),
    )
    result = Study(problem, schedule=Schedule.default(max_iter=60, tol=1e-3)).run()
    out = IMG / "michell.png"
    fig = cast("Figure", result.view())
    fig.savefig(out, dpi=150, bbox_inches="tight")
    _report("michell", out, result)
    return out


def cantilever_3d() -> Path:
    """3D cantilever, 24x12x12: left face fully fixed, downward load at the tip face centre."""
    nelx, nely, nelz = 24, 12, 12
    mesh = StructuredGrid.box(
        size=(float(nelx), float(nely), float(nelz)), shape=(nelx, nely, nelz)
    )
    left = PlaneSlab(point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), tol=1e-9)
    tip = NearPoint((float(nelx), nely / 2.0, nelz / 2.0))
    model = LinearElasticity(
        mesh,
        Material(E=1.0, nu=0.3, rho=1.0),
        supports=[(left, "all")],
        loads=[PointLoad(tip, force=(0.0, -1.0, 0.0))],
    )
    chain = RadialDensityFilter(radius=1.5) | SIMP(p=3.0)
    problem = Problem(
        model,
        chain,
        objective=Compliance(),
        constraints=[Volume() <= 0.3],
        optimizer=OC(move=0.2),
    )
    result = Study(problem, schedule=Schedule.default(max_iter=60, tol=1e-3)).run()
    out = IMG / "cantilever-3d.png"
    plotter = cast("Plotter", result.view(off_screen=True))
    plotter.screenshot(str(out))
    _report("cantilever_3d", out, result)
    return out


# --- Thermal case: heat conduction, developed in docs/content/extend/custom-physics.md.
# RampConductivity, _BoundRamp, and HeatConduction below are reused verbatim from that
# page; see it for the derivation and the gradient check through model and solver.

CONDUCTIVITY = FieldSpec("conductivity_scale")


@dataclass(frozen=True)
class RampConductivity(LinkSpec):
    """RAMP-style terminal interpolation for conduction."""

    q: float = 3.0
    scale_min: float = 1e-3
    is_terminal: ClassVar[bool] = True

    def build(self, mesh: Mesh) -> _BoundRamp:
        return _BoundRamp(self.q, self.scale_min)


class _BoundRamp(BoundLink):
    out_field = CONDUCTIVITY

    def __init__(self, q: float, smin: float) -> None:
        self.q, self.smin = q, smin

    def apply(self, x: _F64) -> _F64:
        return self.smin + (1.0 - self.smin) * x / (1.0 + self.q * (1.0 - x))

    def pullback(self, x: _F64, g: _F64) -> _F64:
        d = (1.0 + self.q) / (1.0 + self.q * (1.0 - x)) ** 2
        return g * (1.0 - self.smin) * d


class HeatConduction:
    """2D conduction on a structured grid: temperature fixed on the left edge,
    a unit heat sink at mid-right, one degree of freedom per node."""

    expected_field: ClassVar[FieldSpec] = CONDUCTIVITY

    def __init__(self, mesh: StructuredGrid) -> None:
        self._mesh = mesh
        hx, hy = mesh.spacing
        r = hx / hy
        a, b = r / 3.0, 1.0 / (3.0 * r)
        self._ke = np.array(
            [
                [a + b, -a + b / 2, -a / 2 - b / 2, a / 2 - b],
                [-a + b / 2, a + b, a / 2 - b, -a / 2 - b / 2],
                [-a / 2 - b / 2, a / 2 - b, a + b, -a + b / 2],
                [a / 2 - b, -a / 2 - b / 2, -a + b / 2, a + b],
            ]
        )
        nx = mesh.shape[0]
        fixed = np.zeros(mesh.n_nodes, dtype=bool)
        fixed[:: nx + 1] = True
        self._free = np.full(mesh.n_nodes, -1, dtype=np.int64)
        self._free[~fixed] = np.arange((~fixed).sum())
        self._n_dof = int((~fixed).sum())
        edofs = self._free[mesh.element_nodes]
        rows = np.repeat(edofs, 4, axis=1).ravel()
        cols = np.tile(edofs, (1, 4)).ravel()
        keep = (rows >= 0) & (cols >= 0)
        self._rows, self._cols, self._keep = rows[keep], cols[keep], keep
        f = np.zeros(self._n_dof)
        sink = int(nx + (nx + 1) * (mesh.shape[1] // 2))
        f[self._free[sink]] = 1.0
        self._loads = f[:, None]

    @property
    def mesh(self) -> StructuredGrid:
        return self._mesh

    @property
    def n_dof(self) -> int:
        return self._n_dof

    @property
    def n_cases(self) -> int:
        return 1

    def assemble(self, scale: _F64) -> SparseMatrix:
        vals = np.einsum("e,k->ek", np.asarray(scale), self._ke.ravel()).ravel()[self._keep]
        return active_backend().coo_to_csr(
            self._rows, self._cols, vals, shape=(self._n_dof, self._n_dof)
        )

    def loads(self) -> _F64:
        return self._loads

    def element_energies(self, u: _F64, scale: _F64) -> _F64:
        uu = np.append(np.asarray(u), 0.0)
        idx = np.where(
            self._free[self._mesh.element_nodes] >= 0,
            self._free[self._mesh.element_nodes],
            self._n_dof,
        )
        ue = uu[idx]
        result: _F64 = np.einsum("ei,ij,ej->e", ue, self._ke, ue) * np.asarray(scale)
        return result

    def element_stress(self, u: _F64, scale: _F64) -> _F64:
        return np.zeros(self._mesh.n_elements)


def thermal() -> Path:
    """Thermal conduction, 40x40: temperature fixed on the left edge, unit heat sink
    at mid-right, RAMP-interpolated conductivity."""
    mesh = StructuredGrid.box(size=(40.0, 40.0), shape=(40, 40))
    problem = Problem(
        HeatConduction(mesh),
        DensityFilter(radius=1.5) | RampConductivity(),
        objective=Compliance(),
        constraints=[Volume() <= 0.3],
        optimizer=MMA(),
    )
    result = Study(problem, schedule=Schedule.default(max_iter=60, tol=1e-3)).run()
    out = IMG / "thermal.png"
    fig = cast("Figure", result.view())
    fig.savefig(out, dpi=150, bbox_inches="tight")
    _report("thermal", out, result)
    return out


def home_hero() -> Path:
    """The first-2D tutorial's 60x20 cantilever, saved as the docs home hero image."""
    mesh = StructuredGrid.box(size=(60.0, 20.0), shape=(60, 20))
    model = LinearElasticity(
        mesh,
        Material(E=1.0, nu=0.3, rho=1.0),
        supports=[(PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0)), "all")],
        loads=[PointLoad(NearPoint((60.0, 10.0)), force=(0.0, -1.0))],
    )
    problem = Problem(
        model,
        DensityFilter(radius=1.5) | SIMP(p=3.0),
        objective=Compliance(),
        constraints=[Volume() <= 0.4],
        optimizer=OC(),
    )
    result = Study(problem, schedule=Schedule.single(p=3.0, max_iter=60, tol=1e-3)).run()
    out = ASSETS / "home-cantilever.png"
    fig = cast("Figure", result.view())
    fig.savefig(out, dpi=150, bbox_inches="tight")
    _report("home_hero", out, result)
    return out


def main() -> None:
    """Regenerate all six gallery/hero images and print their paths."""
    for case in (mbb, cantilever, michell, cantilever_3d, thermal, home_hero):
        out = case()
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
