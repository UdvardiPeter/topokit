# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Lazy dependency loading and display detection for topokit.viz.

The heavy deps (PyVista, matplotlib) live behind the ``[viz]`` extra; import
them only when a view is actually built, and fail with an actionable message.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from types import ModuleType

_log = logging.getLogger("topokit.viz")

VIZ_HINT = "topokit visualization needs the [viz] extra: pip install topokit[viz]"


def require_matplotlib() -> ModuleType:
    """Import matplotlib or raise with the ``[viz]`` install hint."""
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(VIZ_HINT) from exc
    return matplotlib


def require_pyvista() -> ModuleType:
    """Import PyVista or raise with the ``[viz]`` install hint."""
    # import_module rather than a plain import: it is declared to return a
    # ModuleType, so this typechecks the same whether or not the [viz] extra
    # is installed in the checking environment
    try:
        return importlib.import_module("pyvista")
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(VIZ_HINT) from exc


def has_display() -> bool:
    """Best-effort check for an interactive display (for live/interactive views)."""
    if sys.platform in ("darwin", "win32"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
