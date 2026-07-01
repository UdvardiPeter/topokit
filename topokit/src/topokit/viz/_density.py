# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Density-field views: 2D heatmap (matplotlib), 3D iso-surface (PyVista)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from topokit.viz._backend import has_display, require_matplotlib, require_pyvista

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from pyvista import Plotter


class VizError(ValueError):
    """A view was asked for something it cannot render (e.g. slices of a 2D field)."""


def _resolve(obj: Any) -> tuple[np.ndarray, Any]:
    """Extract ``(values, mesh)`` from a Result, DesignField, or the field itself."""
    design = getattr(obj, "design", obj)  # Result.design, else obj is a DesignField
    return np.asarray(design.values, dtype=np.float64), design.mesh


def _image_data(values: np.ndarray, mesh: Any) -> Any:
    """Build a PyVista ImageData with density as cell data (x-fastest = VTK order)."""
    pv = require_pyvista()
    dims = tuple(n + 1 for n in mesh.shape)  # point dims = cells + 1 per axis
    grid = pv.ImageData(dimensions=dims, spacing=tuple(mesh.spacing), origin=tuple(mesh.origin))
    grid.cell_data["rho"] = values
    return grid


def _isosurface(values: np.ndarray, mesh: Any, iso: float) -> Any:
    """Density iso-surface at ``iso`` (cell data → point data → contour)."""
    return _image_data(values, mesh).cell_data_to_point_data().contour([iso], scalars="rho")


def view(obj: Any, *, iso: float = 0.5, off_screen: bool = False) -> Figure | Plotter:
    """Render a density field: 2D heatmap or 3D iso-surface at ``iso``.

    2D returns a matplotlib ``Figure`` (display inline in Jupyter or ``fig.savefig``);
    3D returns a PyVista ``Plotter`` (``.show()`` for a window, ``.screenshot()`` for
    a file). Pass ``off_screen=True`` to render the 3D scene without a window.
    """
    values, mesh = _resolve(obj)
    if mesh.dim == 2:
        require_matplotlib()
        from matplotlib.figure import Figure

        fig = Figure()
        ax = fig.subplots()
        ax.imshow(mesh.to_grid(values).T, origin="lower", cmap="gray_r", vmin=0.0, vmax=1.0)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return fig
    pv = require_pyvista()
    plotter: Plotter = pv.Plotter(off_screen=off_screen or not has_display())
    plotter.add_mesh(_isosurface(values, mesh, iso), color="tan")
    return plotter


_AXES = {"x": 0, "y": 1, "z": 2}


def view_slices(obj: Any, *, axis: str = "z", n: int = 3) -> Figure:
    """3D density as a grid of ``n`` cross-sections along ``axis`` (matplotlib).

    The interactive PyVista plane-widget variant is deferred to a later WP; this
    v1 renders the static, headless-testable small-multiples grid.
    """
    values, mesh = _resolve(obj)
    if mesh.dim != 3:
        raise VizError("view_slices needs a 3D field; a 2D field is already a slice — use view()")
    if axis not in _AXES:
        raise VizError(f"axis must be one of {sorted(_AXES)}; got {axis!r}")
    if n < 1:
        raise VizError(f"n must be >= 1; got {n}")
    require_matplotlib()
    from matplotlib.figure import Figure

    grid = mesh.to_grid(values)  # (nx, ny, nz)
    ax_idx = _AXES[axis]
    count = grid.shape[ax_idx]
    positions = np.linspace(0, count - 1, n, dtype=int)
    fig = Figure()
    axes = np.atleast_1d(fig.subplots(1, n))
    for ax, pos in zip(axes, positions, strict=True):
        plane = np.take(grid, int(pos), axis=ax_idx)  # 2D slab
        ax.imshow(plane.T, origin="lower", cmap="gray_r", vmin=0.0, vmax=1.0)
        ax.set_title(f"{axis}={pos}")
    return fig
