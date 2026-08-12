# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Open-source topology optimization for engineers."""

from topokit.backend import active_backend, default_backend, use_backend
from topokit.events import EventBus
from topokit.fem import (
    ABS,
    ALUMINUM_6061,
    PA12,
    RESIN_SLA,
    STEEL,
    BodyForce,
    LinearElasticity,
    Material,
    PointLoad,
    SurfaceTraction,
)
from topokit.fields import FieldSpec
from topokit.mesh import Mesh, StructuredGrid
from topokit.optimizers import MMA, OC, StepResult
from topokit.parametrization import (
    SIMP,
    BoundLink,
    DensityFilter,
    Heaviside,
    LinkSpec,
    RadialDensityFilter,
    SensitivityFilter,
    SymmetryMap,
)
from topokit.problem import IterationState, Problem, Result, Schedule, Stage, Study
from topokit.registry import registry
from topokit.responses import Compliance, Constraint, ResponseBase, Solution, Volume, von_mises
from topokit.selection import (
    Box,
    Cylinder,
    FaceSetSelector,
    NearPoint,
    OnBoundary,
    PlaneSlab,
    Predicate,
    Sphere,
)
from topokit.solvers import AmgCG, Direct

__version__ = "0.0.1.dev0"

__all__ = [
    "ABS",
    "ALUMINUM_6061",
    "MMA",
    "OC",
    "PA12",
    "RESIN_SLA",
    "SIMP",
    "STEEL",
    "AmgCG",
    "BodyForce",
    "BoundLink",
    "Box",
    "Compliance",
    "Constraint",
    "Cylinder",
    "DensityFilter",
    "Direct",
    "EventBus",
    "FaceSetSelector",
    "FieldSpec",
    "Heaviside",
    "IterationState",
    "LinearElasticity",
    "LinkSpec",
    "Material",
    "Mesh",
    "NearPoint",
    "OnBoundary",
    "PlaneSlab",
    "PointLoad",
    "Predicate",
    "Problem",
    "RadialDensityFilter",
    "ResponseBase",
    "Result",
    "Schedule",
    "SensitivityFilter",
    "Solution",
    "Sphere",
    "Stage",
    "StepResult",
    "StructuredGrid",
    "Study",
    "SurfaceTraction",
    "SymmetryMap",
    "Volume",
    "active_backend",
    "default_backend",
    "registry",
    "use_backend",
    "von_mises",
]

# Convention: the backends group stores instances; component groups
# (physics, optimizers, solvers, ...) store classes.
registry.register("backends", "numpy", default_backend(), source="topokit.backend")
registry.register("physics", "linear_elasticity", LinearElasticity, source="topokit.fem")
registry.register("solvers", "direct", Direct, source="topokit.solvers")
registry.register("solvers", "amg_cg", AmgCG, source="topokit.solvers")
registry.register("chain_links", "symmetry", SymmetryMap, source="topokit.parametrization")
registry.register("chain_links", "density_filter", DensityFilter, source="topokit.parametrization")
registry.register(
    "chain_links", "radial_density_filter", RadialDensityFilter, source="topokit.parametrization"
)
registry.register("chain_links", "heaviside", Heaviside, source="topokit.parametrization")
registry.register("chain_links", "simp", SIMP, source="topokit.parametrization")
registry.register(
    "chain_links", "sensitivity_filter", SensitivityFilter, source="topokit.parametrization"
)
registry.register("responses", "compliance", Compliance, source="topokit.responses")
registry.register("responses", "volume", Volume, source="topokit.responses")
registry.register("optimizers", "oc", OC, source="topokit.optimizers")
registry.register("optimizers", "mma", MMA, source="topokit.optimizers")
