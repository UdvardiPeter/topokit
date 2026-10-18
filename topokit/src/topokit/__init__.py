# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Open-source topology optimization for engineers."""

from topokit.backend import (
    ArrayBackend,
    KernelError,
    SparseMatrix,
    active_backend,
    default_backend,
    use_backend,
)
from topokit.events import EventBus
from topokit.exceptions import TopoKitError
from topokit.fem import (
    ABS,
    ALUMINUM_6061,
    PA12,
    RESIN_SLA,
    STEEL,
    BodyForce,
    FemError,
    LinearElasticity,
    Material,
    PhysicsModel,
    PointLoad,
    SurfaceTraction,
    UnstructuredElasticity,
)
from topokit.fields import DesignField, ElementField, FieldError, FieldSpec, NodeField
from topokit.mesh import Mesh, MeshError, StructuredGrid, UnstructuredMesh
from topokit.optimizers import MMA, OC, Optimizer, OptimizerError, StepResult
from topokit.parametrization import (
    SIMP,
    BoundLink,
    DensityFilter,
    Heaviside,
    LinkSpec,
    ParametrizationError,
    RadialDensityFilter,
    SensitivityFilter,
    SymmetryMap,
)
from topokit.problem import (
    IterationState,
    Problem,
    ProblemError,
    Result,
    Schedule,
    Stage,
    Study,
)
from topokit.registry import RegistryError, registry
from topokit.responses import (
    Compliance,
    Constraint,
    Response,
    ResponseBase,
    ResponseError,
    Solution,
    Volume,
    von_mises,
)
from topokit.selection import (
    Box,
    Cylinder,
    FaceSetSelector,
    NearPoint,
    OnBoundary,
    PlaneSlab,
    Predicate,
    SelectionError,
    Selector,
    SelectorBase,
    Sphere,
)
from topokit.solvers import AmgCG, Direct, LinearSolver, SolverError
from topokit.testing import GradientMismatchError

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
    "ArrayBackend",
    "BodyForce",
    "BoundLink",
    "Box",
    "Compliance",
    "Constraint",
    "Cylinder",
    "DensityFilter",
    "DesignField",
    "Direct",
    "ElementField",
    "EventBus",
    "FaceSetSelector",
    "FemError",
    "FieldError",
    "FieldSpec",
    "GradientMismatchError",
    "Heaviside",
    "IterationState",
    "KernelError",
    "LinearElasticity",
    "LinearSolver",
    "LinkSpec",
    "Material",
    "Mesh",
    "MeshError",
    "NearPoint",
    "NodeField",
    "OnBoundary",
    "Optimizer",
    "OptimizerError",
    "ParametrizationError",
    "PhysicsModel",
    "PlaneSlab",
    "PointLoad",
    "Predicate",
    "Problem",
    "ProblemError",
    "RadialDensityFilter",
    "RegistryError",
    "Response",
    "ResponseBase",
    "ResponseError",
    "Result",
    "Schedule",
    "SelectionError",
    "Selector",
    "SelectorBase",
    "SensitivityFilter",
    "Solution",
    "SolverError",
    "SparseMatrix",
    "Sphere",
    "Stage",
    "StepResult",
    "StructuredGrid",
    "Study",
    "SurfaceTraction",
    "SymmetryMap",
    "TopoKitError",
    "UnstructuredElasticity",
    "UnstructuredMesh",
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
registry.register(
    "physics",
    "unstructured_elasticity",
    UnstructuredElasticity,
    source="topokit.fem",
)
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
