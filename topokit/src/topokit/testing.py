# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Finite-difference gradient verification.

Checks analytic gradients against central differences. Public so plugins
can use the same checks in their tests.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from topokit.exceptions import TopoKitError

_F64 = npt.NDArray[np.float64]


class GradientMismatchError(TopoKitError, AssertionError):
    """An analytic gradient disagrees with central differences."""


# scipy.differentiate.jacobian was measured against all 20 call sites of
# assert_gradient_matches and is not used here. At its default initial_step=0.5
# every one of the 17 real pullbacks raises: a chain input is a density in
# [0, 1] and half a unit leaves the domain. At initial_step=0.1, the step that
# measured best, it is ~250x more accurate for 1.6x the evaluations, but
# test_heaviside_continuation_endpoint_fd (a saturated Heaviside at beta=32)
# still reports status -2, max iterations, and smaller steps trip more call
# sites, not fewer. A gradient check that reports non-convergence on a
# pullback the sweep verifies to 2e-08 is not a better check.
def _central_difference(f: Callable[[_F64], float], x: _F64, h: float) -> _F64:
    """Dense central-difference gradient of ``f`` at ``x`` with step ``h``."""
    g = np.empty_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2.0 * h)
    return g


def assert_gradient_matches(
    f: Callable[[_F64], float],
    grad: Callable[[_F64], _F64],
    x: _F64,
    *,
    steps: tuple[float, ...] = (1e-3, 1e-4, 1e-5, 1e-6),
    rtol: float = 1e-5,
) -> None:
    """Assert ``grad`` matches central differences over a step-size sweep.

    Only the minimum error over the sweep is compared against ``rtol``,
    since truncation error dominates large steps and round-off dominates
    small ones. Steps are scaled by ``max(1, |x|_inf)``.

    Raises
    ------
    GradientMismatchError
        When the minimum relative error exceeds ``rtol``. The message
        includes the per-step error table.
    """
    x = np.asarray(x, dtype=np.float64)
    analytic = np.asarray(grad(x), dtype=np.float64)
    scale = max(1.0, float(np.abs(x).max()))
    norm = max(float(np.linalg.norm(analytic)), 1e-30)
    errors = {
        h: float(np.linalg.norm(_central_difference(f, x, h * scale) - analytic)) / norm
        for h in steps
    }
    if min(errors.values()) > rtol:
        table = ", ".join(f"h={h:g}: {e:.3e}" for h, e in errors.items())
        raise GradientMismatchError(
            f"min rel error {min(errors.values()):.3e} > rtol {rtol:g} ({table})"
        )
