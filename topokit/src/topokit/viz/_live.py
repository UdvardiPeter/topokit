# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Live density view driven by the event bus (headless-safe, main-thread)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from topokit.events import Event, FieldSnapshot
from topokit.viz._backend import has_display

if TYPE_CHECKING:
    from topokit.problem import Study

_log = logging.getLogger("topokit.viz")


class LiveView:
    """Render the design field live as ``FieldSnapshot`` events arrive.

    Rendering is on the main thread inside the callback; cadence is throttled
    upstream by ``Study.snapshot_every``. With no display it is a silent no-op.
    """

    def __init__(self, *, iso: float = 0.5) -> None:
        self.iso = iso
        self.rendered = 0
        self._enabled = has_display()
        self._plotter: Any = None
        if not self._enabled:
            _log.info("LiveView: no display detected; running as a no-op")

    @classmethod
    def attach(cls, study: Study, *, iso: float = 0.5) -> LiveView:
        """Construct a LiveView and subscribe it to ``study.events``."""
        view = cls(iso=iso)
        study.events.subscribe(FieldSnapshot, view)
        return view

    def __call__(self, event: Event) -> None:
        """Event-bus entry point (EventBus catches and logs any exception)."""
        if not self._enabled or not isinstance(event, FieldSnapshot):
            return
        self._draw(event)
        self.rendered += 1

    def _draw(self, snap: FieldSnapshot) -> None:
        import numpy as np

        mesh = snap.mesh
        values = np.asarray(snap.rho, dtype=np.float64)
        if mesh.dim == 2:  # type: ignore[attr-defined]
            self._draw_2d(mesh, values)
        else:
            self._draw_3d(mesh, values)

    def _draw_2d(self, mesh: Any, values: Any) -> None:
        from topokit.viz._backend import require_matplotlib

        plt = require_matplotlib().pyplot
        if self._plotter is None:
            self._plotter = plt.subplots()
        _fig, ax = self._plotter
        ax.clear()
        ax.imshow(mesh.to_grid(values).T, origin="lower", cmap="gray_r", vmin=0.0, vmax=1.0)
        plt.pause(0.001)

    def _draw_3d(self, mesh: Any, values: Any) -> None:
        from topokit.viz._backend import require_pyvista
        from topokit.viz._density import _image_data

        pv = require_pyvista()
        grid = _image_data(values, mesh).cell_data_to_point_data()
        surface = grid.contour([self.iso], scalars="rho")
        if self._plotter is None:
            self._plotter = pv.Plotter()
            self._plotter.show(interactive_update=True, auto_close=False)
        self._plotter.clear()
        self._plotter.add_mesh(surface, color="tan")
        self._plotter.update()
