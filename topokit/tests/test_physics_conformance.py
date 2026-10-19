# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Conformance suite the PhysicsModel contract requires of every element family.

Runs identical checks against `LinearElasticity` (quad4) and
`UnstructuredElasticity` (tri3, tet4, tet10). The two model files each
carried their own copy of these -- four under the same name, seven more
under a suffixed one -- and `test_fem_unstructured.py`'s docstring said it
was "mirroring the grid model's bar". This is that bar, written once.

Every fixture is the same physics problem: a bar along x, rollers on the
x=0 face plus the point pins that remove the remaining rigid modes, and a
uniform traction `SIGMA` on the far face. The exact answer is uniform
`sigma_xx = SIGMA` and a tip displacement of `SIGMA * length / E`, which
every element family here represents exactly, so the tolerances are
solver-precision rather than mesh-dependent.

What is *not* here: anything `ElasticityBase` derives for every subclass
(scale validation, the read-only load matrix, the load-case forms). Those
are checked once, against one subclass, the way `test_mesh_conformance.py`
checks `MeshBase`. What is here either runs through the `_build_elements`
hook or reads a number the element quadrature produced.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from tests.conftest import (
    LEFT_EDGE_2D,
    LEFT_EDGE_3D,
    UNIT_MATERIAL,
    ones,
    solve,
    tet4_box,
    tet10_box,
    tri3_square,
)
from topokit.fem import (
    ElasticityBase,
    FemError,
    LinearElasticity,
    PhysicsModel,
    SurfaceTraction,
    UnstructuredElasticity,
)
from topokit.fields import FieldSpec
from topokit.mesh import Mesh, StructuredGrid, UnstructuredMesh
from topokit.registry import registry
from topokit.selection import Box, NearPoint, Selector

SIGMA = 10.0

Supports = Sequence[tuple[Selector, str | Sequence[str | int]]]

# Roller on x=0 plus the point pins that leave exactly one solvable state:
# in 2D one y-pin, in 3D a y/z-pin at the origin and a second y-pin off the
# z axis to kill the rotation about x.
_SUPPORTS_2D: Supports = [(LEFT_EDGE_2D, "x"), (NearPoint((0.0, 0.0)), "y")]
_SUPPORTS_3D: Supports = [
    (LEFT_EDGE_3D, "x"),
    (NearPoint((0.0, 0.0, 0.0)), ("y", "z")),
    (NearPoint((0.0, 0.0, 1.0)), ("y",)),
]


def _traction(hi: tuple[float, ...]) -> SurfaceTraction:
    """Uniform +x traction on the face at x = hi[0]."""
    dim = len(hi)
    face = Box((hi[0],) + (0.0,) * (dim - 1), hi, tol=1e-9)
    return SurfaceTraction(face, traction=(SIGMA,) + (0.0,) * (dim - 1))


def _with_kind(mesh: UnstructuredMesh, kind: str) -> UnstructuredMesh:
    """``mesh`` reporting ``kind`` as its element kind, geometry untouched."""

    class FakeKindMesh(UnstructuredMesh):
        @property
        def element_kind(self) -> str:
            return kind

    return FakeKindMesh(mesh.nodes, mesh.element_nodes, mesh.element_kind)


def _grid_bar(
    mesh: StructuredGrid | None = None,
    supports: Supports = _SUPPORTS_2D,
    mode: str = "plane_stress",
) -> ElasticityBase[Any]:
    return LinearElasticity(
        StructuredGrid(shape=(4, 2), spacing=(1.0, 1.0)) if mesh is None else mesh,
        UNIT_MATERIAL,
        supports=supports,
        loads=[_traction((4.0, 2.0))],
        mode=mode,
    )


def _grid_with_kind(kind: str) -> StructuredGrid:
    class FakeKindGrid(StructuredGrid):
        @property
        def element_kind(self) -> str:
            return kind

    return FakeKindGrid(shape=(4, 2), spacing=(1.0, 1.0))


def _tri3_bar(
    mesh: Mesh | None = None,
    supports: Supports = _SUPPORTS_2D,
    mode: str = "plane_stress",
) -> ElasticityBase[Any]:
    return UnstructuredElasticity(
        tri3_square() if mesh is None else mesh,
        UNIT_MATERIAL,
        supports=supports,
        loads=[_traction((1.0, 1.0))],
        mode=mode,
    )


def _tet4_bar(
    mesh: Mesh | None = None,
    supports: Supports = _SUPPORTS_3D,
    mode: str = "plane_stress",
) -> ElasticityBase[Any]:
    return UnstructuredElasticity(
        tet4_box(1, 1, 1) if mesh is None else mesh,
        UNIT_MATERIAL,
        supports=supports,
        loads=[_traction((1.0, 1.0, 1.0))],
        mode=mode,
    )


def _tet10_bar(
    mesh: Mesh | None = None,
    supports: Supports = _SUPPORTS_3D,
    mode: str = "plane_stress",
) -> ElasticityBase[Any]:
    return UnstructuredElasticity(
        tet10_box(1, 1, 1) if mesh is None else mesh,
        UNIT_MATERIAL,
        supports=supports,
        loads=[_traction((1.0, 1.0, 1.0))],
        mode=mode,
    )


_TRI3_AS_QUAD4 = _with_kind(tri3_square(), "quad4")
_TET4_AS_HEX8 = _with_kind(tet4_box(1, 1, 1), "hex8")
_TET10_AS_QUAD4 = _with_kind(tet10_box(1, 1, 1), "quad4")


@dataclass(frozen=True)
class Fixture:
    """A `PhysicsModel` on one element family, plus what the bar must produce."""

    id: str
    build: Callable[..., ElasticityBase[Any]]
    dim: int
    length: float  # the bar's x-extent: the exact tip u_x is SIGMA * length / E
    registry_name: str
    bad_kind: str  # an element kind this family must refuse
    bad_kind_mesh: Mesh


FIXTURES = [
    Fixture("quad4_grid", _grid_bar, 2, 4.0, "linear_elasticity", "tet4", _grid_with_kind("tet4")),
    Fixture("tri3", _tri3_bar, 2, 1.0, "unstructured_elasticity", "quad4", _TRI3_AS_QUAD4),
    Fixture("tet4", _tet4_bar, 3, 1.0, "unstructured_elasticity", "hex8", _TET4_AS_HEX8),
    Fixture("tet10", _tet10_bar, 3, 1.0, "unstructured_elasticity", "quad4", _TET10_AS_QUAD4),
]


@pytest.mark.parametrize("fx", FIXTURES, ids=[f.id for f in FIXTURES])
class TestPhysicsConformance:
    """Every `PhysicsModel` implementation must satisfy these checks identically."""

    def test_satisfies_physics_protocol(self, fx: Fixture) -> None:
        model = fx.build()
        assert isinstance(model, PhysicsModel)
        assert model.expected_field == FieldSpec("stiffness_scale")
        assert model.n_cases == 1
        assert type(model) is registry.get("physics", fx.registry_name)

    def test_uniaxial_patch_is_exact(self, fx: Fixture) -> None:
        model = fx.build()
        scale = ones(model.mesh)
        u = solve(model, scale)
        np.testing.assert_allclose(model.element_stress(u, scale), SIGMA, rtol=1e-9)
        tip = np.isclose(model.mesh.nodes[:, 0], fx.length)
        values = model.displacement_field(u).values[tip, 0]
        np.testing.assert_allclose(values, SIGMA * fx.length / UNIT_MATERIAL.E, rtol=1e-9)

    def test_energy_identity(self, fx: Fixture) -> None:
        model = fx.build()
        scale = ones(model.mesh)
        u = solve(model, scale)
        energy = float(u @ model.assemble(scale).matvec(u))
        assert energy == pytest.approx(float(model.loads()[:, 0] @ u), rel=1e-9)
        assert model.element_energies(u, scale).sum() == pytest.approx(energy, rel=1e-9)

    def test_near_nullspace_spans_the_rigid_body_modes(self, fx: Fixture) -> None:
        model = fx.build()
        n_modes = 3 if fx.dim == 2 else 6
        b = model.near_nullspace()
        assert b.shape == (model.n_dof, n_modes)
        assert np.linalg.matrix_rank(b) == n_modes

    def test_support_selecting_nothing_raises(self, fx: Fixture) -> None:
        nowhere = Box((50.0,) * fx.dim, (51.0,) * fx.dim, tol=0.0)
        with pytest.raises(FemError, match="no active nodes"):
            fx.build(supports=[(nowhere, "all")])

    def test_unsupported_element_kind_rejected(self, fx: Fixture) -> None:
        with pytest.raises(FemError, match=fx.bad_kind):
            fx.build(mesh=fx.bad_kind_mesh)

    def test_plane_mode_is_2d_only(self, fx: Fixture) -> None:
        # the mode is a 2D constitutive choice; asking for it in 3D is a
        # modeling mistake, not something to ignore
        if fx.dim == 3:
            with pytest.raises(FemError, match="2D"):
                fx.build(mode="plane_strain")
        else:
            assert fx.build(mode="plane_strain").n_dof > 0
