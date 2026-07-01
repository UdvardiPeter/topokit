# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Visualization for TopoKit runs (extra: ``[viz]``).

A pure event/data consumer: free functions render a ``Result``/``DesignField``;
``LiveView`` subscribes to the event bus. matplotlib for curves/2D/slices,
PyVista for 3D iso-surfaces. Everything is headless-safe.
"""

from __future__ import annotations

from topokit.viz._convergence import plot_convergence

__all__ = ["plot_convergence"]
