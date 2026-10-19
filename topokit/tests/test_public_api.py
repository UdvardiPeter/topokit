# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""The user-facing vocabulary is importable from the top-level package."""

import importlib
import inspect
import subprocess
import sys

import topokit

# Everything the canonical scripted example touches: mesh, physics,
# materials, loads, selectors, chain links, responses, optimizers, orchestration.
USER_FACING = [
    # mesh
    "StructuredGrid",
    "UnstructuredMesh",
    # fem: model, materials, loads
    "LinearElasticity",
    "UnstructuredElasticity",
    "Material",
    "STEEL",
    "ALUMINUM_6061",
    "ABS",
    "PA12",
    "RESIN_SLA",
    "PointLoad",
    "SurfaceTraction",
    "BodyForce",
    # selection
    "Box",
    "Sphere",
    "Cylinder",
    "PlaneSlab",
    "NearPoint",
    "OnBoundary",
    "Predicate",
    "FaceSetSelector",
    # parametrization
    "SymmetryMap",
    "DensityFilter",
    "RadialDensityFilter",
    "Heaviside",
    "SIMP",
    "SensitivityFilter",
    # responses
    "Compliance",
    "Volume",
    "Constraint",
    "von_mises",
    # optimizers / solvers
    "OC",
    "MMA",
    "Direct",
    "AmgCG",
    # orchestration + events
    "Problem",
    "Study",
    "Schedule",
    "Result",
    "EventBus",
    # authoring surface
    "BoundLink",
    "FieldSpec",
    "LinkSpec",
    "Mesh",
    "ResponseBase",
    "Solution",
    "StepResult",
    # the protocols the Extend guides tell authors to implement, and that
    # Problem construction names when a component does not satisfy one
    "ArrayBackend",
    "LinearSolver",
    "Optimizer",
    "PhysicsModel",
    "Response",
    "Selector",
    "SelectorBase",
    "SparseMatrix",
    # the field containers the public API hands back
    "DesignField",
    "ElementField",
    "NodeField",
]

# Exception types a caller can catch by name. The meta-test below walks these
# modules, so a new error type that nobody exported fails the suite.
ERROR_MODULES = (
    "topokit.backend",
    "topokit.fem",
    "topokit.fields",
    "topokit.mesh",
    "topokit.optimizers",
    "topokit.parametrization",
    "topokit.problem",
    "topokit.registry",
    "topokit.responses",
    "topokit.selection",
    "topokit.solvers",
    "topokit.testing",
)
# topokit.viz is deliberately absent: VizError lives behind the [viz] extra, and
# exporting it would make `import topokit` pull that namespace in. Catch it as
# topokit.viz.VizError, where its callers already are.


def test_user_facing_names_are_top_level() -> None:
    for name in USER_FACING:
        assert hasattr(topokit, name), f"topokit.{name} missing"
        assert name in topokit.__all__, f"{name} not in __all__"


def test_every_error_type_is_catchable_by_name() -> None:
    for module_name in ERROR_MODULES:
        module = importlib.import_module(module_name)
        for attr, obj in vars(module).items():
            if (
                inspect.isclass(obj)
                and issubclass(obj, Exception)
                and obj.__module__.startswith(module_name)
            ):
                assert attr in topokit.__all__, f"{module_name}.{attr} is not exported"


def test_viz_error_stays_behind_the_extra() -> None:
    assert "VizError" not in topokit.__all__


def test_importing_topokit_does_not_pull_in_the_viz_namespace() -> None:
    # the reason VizError is not exported; a fresh interpreter, because other
    # tests in this session have already imported topokit.viz
    probe = subprocess.run(
        [sys.executable, "-c", "import sys, topokit; print('topokit.viz' in sys.modules)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout.strip() == "False"
