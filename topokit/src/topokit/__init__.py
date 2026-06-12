# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Open-source topology optimization for engineers."""

from topokit.backend import default_backend
from topokit.fem import LinearElasticity
from topokit.registry import registry

__version__ = "0.0.1.dev0"

# Convention: the backends group stores instances; component groups
# (physics, optimizers, solvers, ...) store classes.
registry.register("backends", "numpy", default_backend(), source="topokit.backend")
registry.register("physics", "linear_elasticity", LinearElasticity, source="topokit.fem")
