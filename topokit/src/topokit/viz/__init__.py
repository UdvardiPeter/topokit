# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Visualization for TopoKit runs (extra: ``[viz]``).

A pure event/data consumer: free functions render a ``Result``/``DesignField``;
``LiveView`` subscribes to the event bus. matplotlib for curves/2D/slices,
PyVista for 3D iso-surfaces. Everything is headless-safe.

Display notes:

- The matplotlib returns (``plot_convergence``, 2D ``view``, ``view_slices``) are
  object-oriented ``Figure`` objects — they display inline in Jupyter and save via
  ``fig.savefig(...)``, but are not tied to ``pyplot`` so ``plt.show()`` won't pick
  them up in a plain script.
- 3D ``view`` returns a PyVista ``Plotter``; call ``.show()`` (window) or
  ``.screenshot(...)`` (file). Interactive 3D *inline in Jupyter* needs the trame
  backend, which ``[viz]`` keeps out to stay lean: ``pip install "pyvista[jupyter]"``
  enables it. 2D/curve views and static 3D screenshots need nothing extra.
"""

from __future__ import annotations

from topokit.viz._convergence import plot_convergence
from topokit.viz._density import VizError, view, view_slices
from topokit.viz._live import LiveView

__all__ = ["LiveView", "VizError", "plot_convergence", "view", "view_slices"]
