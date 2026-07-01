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
    grid = pv.ImageData(dimensions=dims, spacing=tuple(mesh.spacing), origin=(0.0, 0.0, 0.0))
    grid.cell_data["rho"] = values
    return grid


def view(obj: Any, *, iso: float = 0.5, off_screen: bool = False) -> Figure | Plotter:
    """Render a density field: 2D heatmap or 3D iso-surface at ``iso``.

    2D returns a matplotlib ``Figure``; 3D returns a PyVista ``Plotter``.
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
    surface = _image_data(values, mesh).cell_data_to_point_data().contour([iso], scalars="rho")
    plotter: Plotter = pv.Plotter(off_screen=off_screen or not has_display())
    plotter.add_mesh(surface, color="tan")
    return plotter
