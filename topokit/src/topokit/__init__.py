# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Open-source topology optimization for engineers."""

from topokit.backend import NumpyBackend
from topokit.registry import registry

__version__ = "0.0.1.dev0"

registry.register("backends", "numpy", NumpyBackend, source="topokit.backend")
