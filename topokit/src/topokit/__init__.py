# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Open-source topology optimization for engineers."""

from topokit.backend import default_backend
from topokit.fem import LinearElasticity
from topokit.optimizers import MMA, OC
from topokit.parametrization import (
    SIMP,
    DensityFilter,
    Heaviside,
    SensitivityFilter,
    SymmetryMap,
)
from topokit.problem import IterationState, Problem, Result, Study
from topokit.registry import registry
from topokit.responses import Compliance, Volume
from topokit.solvers import AmgCG, Direct

__version__ = "0.0.1.dev0"

__all__ = [
    "MMA",
    "OC",
    "SIMP",
    "AmgCG",
    "Compliance",
    "DensityFilter",
    "Direct",
    "Heaviside",
    "IterationState",
    "LinearElasticity",
    "Problem",
    "Result",
    "SensitivityFilter",
    "Study",
    "SymmetryMap",
    "Volume",
    "default_backend",
    "registry",
]

# Convention: the backends group stores instances; component groups
# (physics, optimizers, solvers, ...) store classes.
registry.register("backends", "numpy", default_backend(), source="topokit.backend")
registry.register("physics", "linear_elasticity", LinearElasticity, source="topokit.fem")
registry.register("solvers", "direct", Direct, source="topokit.solvers")
registry.register("solvers", "amg_cg", AmgCG, source="topokit.solvers")
registry.register("chain_links", "symmetry", SymmetryMap, source="topokit.parametrization")
registry.register("chain_links", "density_filter", DensityFilter, source="topokit.parametrization")
registry.register("chain_links", "heaviside", Heaviside, source="topokit.parametrization")
registry.register("chain_links", "simp", SIMP, source="topokit.parametrization")
registry.register(
    "chain_links", "sensitivity_filter", SensitivityFilter, source="topokit.parametrization"
)
registry.register("responses", "compliance", Compliance, source="topokit.responses")
registry.register("responses", "volume", Volume, source="topokit.responses")
registry.register("optimizers", "oc", OC, source="topokit.optimizers")
registry.register("optimizers", "mma", MMA, source="topokit.optimizers")
