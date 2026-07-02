# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""The user-facing vocabulary is importable from the top-level package."""

import topokit

# Everything the canonical scripted example (doc 03) touches: mesh, physics,
# materials, loads, selectors, chain links, responses, optimizers, orchestration.
USER_FACING = [
    # mesh
    "StructuredGrid",
    # fem: model, materials, loads
    "LinearElasticity",
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
]


def test_user_facing_names_are_top_level() -> None:
    for name in USER_FACING:
        assert hasattr(topokit, name), f"topokit.{name} missing"
        assert name in topokit.__all__, f"{name} not in __all__"
