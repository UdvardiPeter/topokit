# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Tier-3 full robustness sweep (pytest -m regression_full).

Checks clean convergence across the parameter space — behaviour, not frozen
values: finite field in [0, 1], volume on target, net objective progress, no
oscillation blow-up.
"""

import numpy as np
import pytest
from topokit.problem import Result, Schedule, Study

from topokit_bench.problems import Builder, cantilever, cantilever_3d, make_optimizer, mbb

_2D = [
    (builder, nelx, nely, volfrac, rmin)
    for builder in (mbb, cantilever)
    for nelx, nely in ((60, 20), (120, 40))
    for volfrac in (0.3, 0.5)
    for rmin in (1.5, 3.0)
]


def _assert_clean(result: Result, volfrac: float) -> None:
    field = np.asarray(result.design.values)
    assert np.all(np.isfinite(field))
    assert field.min() >= -1e-9
    assert field.max() <= 1.0 + 1e-9
    assert np.isfinite(result.objective)
    assert result.history["volume"][-1] == pytest.approx(volfrac, abs=1e-3)
    obj = np.asarray(result.history["objective"], dtype=float)
    assert obj[-1] < obj[0]  # net progress over the stage
    assert np.all(obj <= 5.0 * obj[0])  # no oscillation blow-up


@pytest.mark.regression_full
@pytest.mark.parametrize("builder,nelx,nely,volfrac,rmin", _2D)
def test_2d_converges_cleanly(
    builder: Builder, nelx: int, nely: int, volfrac: float, rmin: float
) -> None:
    problem = builder(nelx, nely, volfrac=volfrac, rmin=rmin, optimizer=make_optimizer("oc"))
    result = Study(problem, schedule=Schedule.single(p=3.0, max_iter=60, tol=1e-3)).run()
    _assert_clean(result, volfrac)


@pytest.mark.regression_full
@pytest.mark.parametrize("nelx,nely,nelz", [(16, 8, 8), (20, 10, 10)])
def test_3d_converges_cleanly(nelx: int, nely: int, nelz: int) -> None:
    problem = cantilever_3d(nelx, nely, nelz, optimizer=make_optimizer("oc"))
    result = Study(problem, schedule=Schedule.single(p=3.0, max_iter=40, tol=1e-3)).run()
    _assert_clean(result, 0.3)


@pytest.mark.regression_full
def test_default_schedule_with_mma_and_projection_converges_cleanly() -> None:
    # Audit item I: the out-of-the-box path (default schedule, MMA defaults,
    # Heaviside projection) destabilized at high beta: the projection saturates
    # most elements, their gradients vanish, and MMA's linear model then sees
    # deleting load-bearing material as free volume slack. The objective spiked
    # ~7 orders of magnitude and the final stage never converged. Guard the
    # whole property: converges, no blowup, ends near the pre-spike optimum.
    from topokit.fem import LinearElasticity, Material, PointLoad
    from topokit.mesh import StructuredGrid
    from topokit.optimizers import MMA
    from topokit.parametrization import SIMP, DensityFilter, Heaviside
    from topokit.problem import Problem
    from topokit.responses import Compliance, Volume
    from topokit.selection import NearPoint, PlaneSlab

    grid = StructuredGrid.box(size=(60.0, 20.0), shape=(60, 20))
    left = PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0), tol=1e-9)
    model = LinearElasticity(
        grid,
        Material(E=1.0, nu=0.3, rho=1.0),
        supports=[(left, "all")],
        loads=[PointLoad(NearPoint((60.0, 10.0)), (0.0, -1.0))],
    )
    chain = DensityFilter(radius=1.5) | Heaviside() | SIMP()
    problem = Problem(
        model, chain, objective=Compliance(), constraints=[Volume() <= 0.4], optimizer=MMA()
    )
    result = Study(problem).run()  # default schedule, all defaults
    obj = np.asarray(result.history["objective"], dtype=float)
    assert result.converged, f"unconverged: {result.reason}"
    assert obj.max() < 3.0 * obj[0], f"objective spiked to {obj.max():.3g}"
    assert result.objective < 1.5 * result.best_objective
