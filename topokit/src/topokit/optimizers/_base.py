# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Optimizer protocol shared by OC and MMA.

An optimizer is a pure stepper: given the current design variables and the
objective/constraint values and gradients (already in design-variable
space, the orchestration layer having routed them through the chain
pullbacks), it returns the next point. It knows nothing of chains, physics,
or solvers.
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


def validate_bounds(n_vars: int, lower: _F64, upper: _F64) -> tuple[_F64, _F64]:
    """Coerce and validate the box bounds for ``setup`` (shape, upper > lower)."""
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if lo.shape != (n_vars,) or hi.shape != (n_vars,):
        raise OptimizerError(f"lower/upper must have shape ({n_vars},)")
    if not np.all(hi > lo):
        raise OptimizerError("upper bound must exceed lower bound for every variable")
    return lo, hi


def check_step_inputs(
    lower: _F64 | None, x: _F64, df0: _F64, g: _F64, dg: _F64
) -> tuple[_F64, _F64, _F64, _F64]:
    """Validate and coerce the common ``step`` inputs (shapes, finiteness)."""
    if lower is None:
        raise OptimizerError("call setup() before step()")
    x = np.asarray(x, dtype=np.float64)
    df0 = np.asarray(df0, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    dg = np.asarray(dg, dtype=np.float64)
    n = lower.size
    if x.shape != (n,):
        raise OptimizerError(f"x shape {x.shape} != ({n},) from setup")
    if df0.shape != (n,):
        raise OptimizerError(f"df0 shape {df0.shape} != ({n},)")
    if g.ndim != 1 or dg.shape != (g.size, n):
        raise OptimizerError(f"dg shape {dg.shape} must be ({g.size}, {n})")
    for name, arr in (("x", x), ("df0", df0), ("g", g), ("dg", dg)):
        if not np.isfinite(arr).all():
            raise OptimizerError(f"{name} contains non-finite values")
    return x, df0, g, dg
