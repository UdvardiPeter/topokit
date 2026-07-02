# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Objective and constraint responses.

A `Response` maps a solved state to a scalar and its gradient with respect
to the field it reads. Two field bases exist (the parametrization split):
``Compliance`` reads the interpolated stiffness scale (``apply``), ``Volume``
reads the physical density (``physical_density``); the orchestration layer
routes each gradient through the matching chain pullback.

``Compliance`` is self-adjoint (its gradient is the negated element strain
energies, no extra solve) and ``Volume`` is explicit, so neither needs an
adjoint solve in v1. ``n_extra_adjoints`` declares the count for the
machinery that lands with stress constraints (v1.x).

Comparisons build constraints: ``Volume() <= 0.3`` returns a `Constraint`
normalized to ``g <= 0`` form for the optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from topokit.fields import ElementField
from topokit.mesh import Mesh

if TYPE_CHECKING:
    from topokit.fem import PhysicsModel

_F64 = npt.NDArray[np.float64]

# Which chain field a response differentiates against; the orchestration layer
# routes "interpolated" through chain.pullback and "density" through
# chain.pullback_density.
FieldBasis = Literal["interpolated", "density"]


class ResponseError(ValueError):
    """Invalid response setup or input."""


@dataclass(frozen=True)
class Solution:
    """The solved state plus the chain's fields, the input every response reads."""

    model: PhysicsModel | None
    mesh: Mesh
    displacements: _F64  # (n_dof, n_cases)
    interpolated: _F64  # (n_elements,) stiffness scale from chain.apply
    density: _F64  # (n_elements,) physical density from chain.physical_density


@runtime_checkable
class Response(Protocol):
    """A scalar of the solved state, differentiable w.r.t. one field basis."""

    name: ClassVar[str]
    field_basis: ClassVar[FieldBasis]
    n_extra_adjoints: ClassVar[int]

    def value(self, solution: Solution) -> float:
        """Return the scalar value."""
        ...

    def grad_field(self, solution: Solution) -> _F64:
        """Return the gradient w.r.t. the ``field_basis`` field, shape ``(n_elements,)``."""
        ...


class ResponseBase:
    """Base giving responses the ``<=`` / ``>=`` constraint operators."""

    def __le__(self, bound: float) -> Constraint:
        return Constraint(self, float(bound), "<=")  # type: ignore[arg-type]

    def __ge__(self, bound: float) -> Constraint:
        return Constraint(self, float(bound), ">=")  # type: ignore[arg-type]


@dataclass(frozen=True)
class Compliance(ResponseBase):
    """Structural compliance ``u^T K u`` (lower is stiffer); the usual objective.

    Multi-load: a weighted sum over load cases (default weight 1.0 each, so a
    single case is unchanged). The value is the sum of element strain energies
    ``u^T K u`` (equal to ``f^T u`` at convergence), the standard topology-
    optimization compliance form and the quantity the gradient differentiates.
    """

    weights: tuple[float, ...] | None = None
    name: ClassVar[str] = "compliance"
    field_basis: ClassVar[FieldBasis] = "interpolated"
    n_extra_adjoints: ClassVar[int] = 0

    def _case_weights(self, n_cases: int) -> _F64:
        if self.weights is None:
            return np.ones(n_cases)
        if len(self.weights) != n_cases:
            raise ResponseError(f"{len(self.weights)} weights for {n_cases} load cases")
        return np.asarray(self.weights, dtype=np.float64)

    def value(self, solution: Solution) -> float:
        """Weighted sum of per-case strain energy ``u^T K u``."""
        model = _require_model(solution)
        u = solution.displacements
        w = self._case_weights(u.shape[1])
        total = sum(
            w[i] * float(model.element_energies(u[:, i], solution.interpolated).sum())
            for i in range(u.shape[1])
        )
        return float(total)

    def grad_field(self, solution: Solution) -> _F64:
        """``dc/d(scale)`` = negated unscaled element energies, weighted over cases."""
        model = _require_model(solution)
        u = solution.displacements
        w = self._case_weights(u.shape[1])
        ones = np.ones_like(solution.interpolated)
        g = np.zeros_like(solution.interpolated)
        for i in range(u.shape[1]):
            g -= w[i] * model.element_energies(u[:, i], ones)
        return g

    @classmethod
    def fd_example(cls) -> Compliance:
        """Instance for the responses FD meta-test."""
        return cls()


@dataclass(frozen=True)
class Volume(ResponseBase):
    """Material volume fraction of a region (default the design region).

    Fraction = sum(rho * v) / sum(v) over the region, so ``volume_fraction``
    means "this fraction of the chosen region is filled". ``region`` is
    ``"design"`` (optimized elements; the usual budget), ``"active"``
    (non-void), or ``"all"``.
    """

    region: str = "design"
    name: ClassVar[str] = "volume"
    field_basis: ClassVar[FieldBasis] = "density"
    n_extra_adjoints: ClassVar[int] = 0

    def _mask(self, mesh: Mesh) -> npt.NDArray[np.bool_]:
        if self.region == "design":
            return mesh.design
        if self.region == "active":
            return mesh.active_elements
        if self.region == "all":
            return np.ones(mesh.n_elements, dtype=bool)
        raise ResponseError(f"unknown region {self.region!r}")

    def value(self, solution: Solution) -> float:
        """Region volume fraction."""
        mesh = solution.mesh
        region = self._mask(mesh)
        v = mesh.element_volumes
        total = float(v[region].sum())
        if total == 0.0:
            raise ResponseError(f"volume region {self.region!r} contains no elements")
        return float((solution.density[region] * v[region]).sum() / total)

    def grad_field(self, solution: Solution) -> _F64:
        """``dV/d(rho)`` = element volume fraction on the region, zero elsewhere."""
        mesh = solution.mesh
        region = self._mask(mesh)
        v = mesh.element_volumes
        g = np.zeros(mesh.n_elements)
        g[region] = v[region] / v[region].sum()
        return g

    @classmethod
    def fd_example(cls) -> Volume:
        """Instance for the responses FD meta-test."""
        return cls()


@dataclass(frozen=True)
class Constraint:
    """A response bounded for the optimizer, normalized to ``g <= 0``."""

    response: Response
    bound: float
    sense: str  # "<=" or ">="
    label: str | None = None

    def __post_init__(self) -> None:
        if self.sense not in ("<=", ">="):
            raise ResponseError(f"sense must be '<=' or '>=', got {self.sense!r}")

    @property
    def field_basis(self) -> FieldBasis:
        """Return the field basis of the underlying response."""
        return self.response.field_basis

    @property
    def report_key(self) -> str:
        """Key for this constraint in events and history (``label`` or the response name).

        Two constraints on the same response (e.g. volume on different regions)
        share a response name; give each a distinct ``label`` so the
        orchestration layer can report them apart.
        """
        return self.label if self.label is not None else self.response.name

    def labeled(self, label: str) -> Constraint:
        """Return a copy reported under ``label``."""
        return replace(self, label=label)

    def _scale(self) -> float:
        # dividing by |bound| (not bound) keeps the inequality direction for
        # negative bounds; for positive bounds this is the familiar
        # value/bound - 1 form.
        sign = 1.0 if self.sense == "<=" else -1.0
        return sign if self.bound == 0.0 else sign / abs(self.bound)

    def value(self, solution: Solution) -> float:
        """Constraint value in ``g <= 0`` form (g = (value - bound) / |bound| for ``<=``)."""
        v = self.response.value(solution)
        return self._scale() * (v - self.bound)

    def grad_field(self, solution: Solution) -> _F64:
        """Gradient of the normalized constraint w.r.t. the response's field basis."""
        return self._scale() * self.response.grad_field(solution)


def von_mises(solution: Solution, name: str = "von_mises") -> ElementField:
    """Per-element von Mises stress as a display field (zero at void)."""
    model = _require_model(solution)
    values = model.element_stress(solution.displacements[:, 0], solution.interpolated)
    return ElementField(values, solution.mesh, name=name)


def _require_model(solution: Solution) -> PhysicsModel:
    if solution.model is None:
        raise ResponseError("this response needs a solved model in the Solution")
    return solution.model
