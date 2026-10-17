# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Tier-3 2D reference regression (pytest -m regression)."""

from pathlib import Path

import numpy as np
import pytest
from topokit.problem import Result, Schedule, Study

from topokit_bench.problems import BUILDERS, CASES, make_optimizer

DATA = Path(__file__).resolve().parent / "data"


def _run(name: str, nelx: int, nely: int, opt_name: str) -> Result:
    problem = BUILDERS[name](nelx, nely, optimizer=make_optimizer(opt_name))
    return Study(problem, schedule=Schedule.single(p=3.0, max_iter=100, tol=1e-3)).run()


@pytest.mark.regression
@pytest.mark.parametrize("name,nelx,nely,opt_name", CASES)
def test_matches_reference(name: str, nelx: int, nely: int, opt_name: str) -> None:
    ref = np.load(DATA / f"{name}_{nelx}x{nely}_{opt_name}.npz")
    result = _run(name, nelx, nely, opt_name)
    # 1. volume held
    assert result.history["volume"][-1] == pytest.approx(float(ref["volume"]), abs=1e-3)
    # 2. compliance within 1% of the frozen reference
    assert result.objective == pytest.approx(float(ref["compliance"]), rel=0.01)
    # 3. density field within a coarse 1e-2 (robust to cross-platform LU/BLAS drift)
    assert float(np.mean(np.abs(result.design.values - ref["density"]))) < 1e-2
    # 4. iteration count within +/- 25%
    assert abs(result.iterations - int(ref["iterations"])) <= 0.25 * int(ref["iterations"]) + 1


# Every geometry in the 2D matrix, derived from CASES rather than listed: a
# hardcoded list leaves a newly added case silently ungated, which is how
# cantilever_tri landed with an MMA reference nothing compared against. Each 2D
# case is frozen for both optimizers, so an OC-only addition fails here loudly
# rather than being skipped.
_GEOMETRIES = sorted({(name, nelx, nely) for name, nelx, nely, _ in CASES})


@pytest.mark.regression
@pytest.mark.parametrize("name,nelx,nely", _GEOMETRIES)
def test_oc_and_mma_agree(name: str, nelx: int, nely: int) -> None:
    # the two optimizers reach the same optimum; checked on the frozen references
    # so the per-PR suite does not pay a second full run per case.
    c_oc = float(np.load(DATA / f"{name}_{nelx}x{nely}_oc.npz")["compliance"])
    c_mma = float(np.load(DATA / f"{name}_{nelx}x{nely}_mma.npz")["compliance"])
    assert abs(c_mma - c_oc) / c_oc < 0.05


@pytest.mark.regression
def test_tri3_vs_quad4_cantilever_agree() -> None:
    # tri3 (cantilever_tri) vs quad4 (cantilever) cantilever compliance at
    # 60x20, read off the frozen OC references, so this pays no extra run and
    # gates the pair whenever either is regenerated rather than every run.
    # Both cases are the same problem bar the discretization: same box, same
    # supports and load point, same plane-stress material, same radial filter
    # radius and SIMP exponent.
    # Measured gap: 2.08%, tri3 the lower. Three small effects, all pushing
    # the same way: constant-strain triangles are stiffer than bilinear quads
    # at equal nodes, the split doubles the design variables, and its fixed
    # diagonal biases the design. Tolerance ~2x the measured gap (the plan's
    # 0.15 cap does not bind). A modeling inconsistency -- a different
    # constitutive mode, filter kernel or load node -- moves this by far more.
    c_quad = float(np.load(DATA / "cantilever_60x20_oc.npz")["compliance"])
    c_tri = float(np.load(DATA / "cantilever_tri_60x20_oc.npz")["compliance"])
    assert abs(c_tri - c_quad) / c_quad < 0.05
