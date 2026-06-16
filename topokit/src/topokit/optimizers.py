# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Design-variable optimizers.

An optimizer is a pure stepper: given the current design variables and the
objective/constraint values and gradients (already in design-variable
space, the orchestration layer having routed them through the chain
pullbacks), it returns the next point. It knows nothing of chains, physics,
or solvers.

``OC`` is the classic optimality-criteria update for a single volume-type
constraint: it scales each variable by a sensitivity ratio and bisects the
Lagrange multiplier until the constraint is met. It is the standard
optimizer of the 88-line density method. General multi-constraint problems
need MMA (WP-1.8b).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

_F64 = npt.NDArray[np.float64]


class OptimizerError(ValueError):
    """Invalid optimizer setup or input."""


@dataclass(frozen=True)
class StepResult:
    """One optimizer step: the next point and the max design change."""

    x_next: _F64
    change: float


@runtime_checkable
class Optimizer(Protocol):
    """A stepper over bounded design variables."""

    def setup(self, n_vars: int, lower: _F64, upper: _F64) -> None:
        """Set the variable count and box bounds (once, before stepping)."""
        ...

    def step(self, x: _F64, f0: float, df0: _F64, g: _F64, dg: _F64) -> StepResult:
        """Advance ``x`` given objective ``f0``/``df0`` and constraints ``g``/``dg``."""
        ...

    def state(self) -> dict[str, Any]:
        """Serializable optimizer state for checkpointing."""
        ...

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore from :meth:`state`."""
        ...


@dataclass
class OC:
    """Optimality-criteria update for compliance with one volume constraint.

    ``move`` is the per-step box move limit; ``eta`` the update exponent
    (0.5 is the 88-line standard). Stateless between steps.
    """

    move: float = 0.2
    eta: float = 0.5
    bisection_tol: float = 1e-4

    def __post_init__(self) -> None:
        if not 0.0 < self.move <= 1.0:
            raise OptimizerError(f"move must be in (0, 1], got {self.move}")
        if self.eta <= 0.0:
            raise OptimizerError(f"eta must be > 0, got {self.eta}")
        self._lower: _F64 | None = None
        self._upper: _F64 | None = None

    def setup(self, n_vars: int, lower: _F64, upper: _F64) -> None:
        """Store the box bounds."""
        self._lower = np.asarray(lower, dtype=np.float64)
        self._upper = np.asarray(upper, dtype=np.float64)
        if self._lower.shape != (n_vars,) or self._upper.shape != (n_vars,):
            raise OptimizerError("lower/upper must have shape (n_vars,)")

    def step(self, x: _F64, f0: float, df0: _F64, g: _F64, dg: _F64) -> StepResult:
        """One OC update; bisects the multiplier so the constraint is active."""
        if self._lower is None or self._upper is None:
            raise OptimizerError("call setup() before step()")
        x = np.asarray(x, dtype=np.float64)
        df0 = np.asarray(df0, dtype=np.float64)
        g = np.asarray(g, dtype=np.float64)
        dg = np.asarray(dg, dtype=np.float64)
        n = self._lower.size
        if x.shape != (n,):
            raise OptimizerError(f"x shape {x.shape} != ({n},) from setup")
        if df0.shape != (n,):
            raise OptimizerError(f"df0 shape {df0.shape} != ({n},)")
        for name, arr in (("x", x), ("df0", df0), ("g", g), ("dg", dg)):
            if not np.isfinite(arr).all():
                raise OptimizerError(f"{name} contains non-finite values")
        if g.shape != (1,) or dg.shape != (1, n):
            raise OptimizerError("OC handles exactly one constraint")
        dgdx = dg[0]
        if bool((dgdx < 0.0).any()):
            raise OptimizerError(
                "OC needs a non-negative constraint gradient (volume-like, "
                "increasing in density); use MMA for general constraints"
            )

        lo = np.maximum(self._lower, x - self.move)
        hi = np.minimum(self._upper, x + self.move)
        # only variables that reduce the objective when increased move up
        numerator = np.maximum(0.0, -df0)

        def candidate(lam: float) -> _F64:
            b = numerator / (lam * dgdx + 1e-30)
            step = x * np.sqrt(b) if self.eta == 0.5 else x * b**self.eta
            return np.asarray(np.clip(step, lo, hi), dtype=np.float64)

        # g is in <= 0 form; g(lam) = g0_value scaled. We need the lambda that
        # drives the realized constraint to its bound. Recover the constraint's
        # linear model from (g, dgdx): realized g(x_new) ~ g + dgdx . (x_new - x).
        def residual(lam: float) -> float:
            xn = candidate(lam)
            return float(g[0] + dgdx @ (xn - x))

        l1, l2 = 1e-9, 1e9
        # ensure the bracket spans a sign change (residual decreasing in lambda)
        while residual(l1) < 0.0 and l1 > 1e-30:
            l1 *= 0.1
        while residual(l2) > 0.0 and l2 < 1e30:
            l2 *= 10.0
        for _ in range(200):
            lmid = 0.5 * (l1 + l2)
            if residual(lmid) > 0.0:
                l1 = lmid
            else:
                l2 = lmid
            if (l2 - l1) <= self.bisection_tol * (1.0 + l2):
                break
        x_next = candidate(0.5 * (l1 + l2))
        return StepResult(x_next=x_next, change=float(np.abs(x_next - x).max()))

    def state(self) -> dict[str, Any]:
        """OC is stateless; returns an empty dict."""
        return {}

    def load_state(self, state: dict[str, Any]) -> None:
        """No-op; OC carries no state."""
        return
