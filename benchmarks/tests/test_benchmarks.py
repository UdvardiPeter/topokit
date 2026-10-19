# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Tier-3 reference regression: the 2D matrix (-m regression) and the full one.

One `test_matches_reference` over both case matrices. The two suites were
the same four assertions at the same tolerances against the same frozen
`.npz` layout, differing only in which matrix they read and which marker
they carried; the marker now rides on the parameter, so `-m regression`
still selects the 2D smalls and `-m regression_full` the nightly 3D +
Michell cases.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from topokit.problem import Schedule, Study

from topokit_bench.problems import BUILDERS, CASES, FULL_CASES, Builder, make_optimizer

DATA = Path(__file__).resolve().parent / "data"

# Every frozen reference, read off the case matrices rather than listed: a
# hardcoded list leaves a newly added case silently ungated, which is how
# cantilever_tri landed with an MMA reference nothing compared against.
_RUNS = [
    pytest.param(
        f"{name}_{nelx}x{nely}_{opt}",
        BUILDERS[name],
        {"nelx": nelx, "nely": nely},
        opt,
        marks=pytest.mark.regression,
        id=f"{name}_{nelx}x{nely}_{opt}",
    )
    for name, nelx, nely, opt in CASES
] + [
    pytest.param(
        f"{case.key}_{opt}",
        case.build,
        dict(case.kwargs),
        opt,
        marks=pytest.mark.regression_full,
        id=f"{case.key}_{opt}",
    )
    for case in FULL_CASES
    for opt in case.optimizers
]


@pytest.mark.parametrize("stem,build,kwargs,opt", _RUNS)
def test_matches_reference(stem: str, build: Builder, kwargs: dict[str, int], opt: str) -> None:
    ref = np.load(DATA / f"{stem}.npz")
    problem = build(**kwargs, optimizer=make_optimizer(opt))
    result = Study(problem, schedule=Schedule.single(p=3.0, max_iter=100, tol=1e-3)).run()
    # 1. volume held
    assert result.history["volume"][-1] == pytest.approx(float(ref["volume"]), abs=1e-3)
    # 2. compliance within 1% of the frozen reference
    assert result.objective == pytest.approx(float(ref["compliance"]), rel=0.01)
    # 3. density field within a coarse 1e-2 (robust to cross-platform LU/BLAS drift)
    assert float(np.mean(np.abs(result.design.values - ref["density"]))) < 1e-2
    # 4. iteration count within +/- 25%
    assert abs(result.iterations - int(ref["iterations"])) <= 0.25 * int(ref["iterations"]) + 1


# Every case frozen for both optimizers, again derived from the matrices. The
# 2D cases are all frozen for both; among the full cases only Michell is.
_AGREEMENT_STEMS = sorted(
    {f"{name}_{nelx}x{nely}" for name, nelx, nely, _ in CASES}
    | {case.key for case in FULL_CASES if {"oc", "mma"} <= set(case.optimizers)}
)


@pytest.mark.regression
@pytest.mark.parametrize("stem", _AGREEMENT_STEMS)
def test_oc_and_mma_agree(stem: str) -> None:
    # the two optimizers reach the same optimum; checked on the frozen references
    # so the suite does not pay a second full run per case.
    c_oc = float(np.load(DATA / f"{stem}_oc.npz")["compliance"])
    c_mma = float(np.load(DATA / f"{stem}_mma.npz")["compliance"])
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


@pytest.mark.regression_full
def test_michell_reference_is_vertically_symmetric() -> None:
    # topology-sanity: the committed Michell reference is mirror-symmetric about
    # the horizontal centreline (symmetric BCs + load).
    ref = np.load(DATA / "michell_90x30_oc.npz")
    field: Any = ref["density"].reshape(30, 90)  # (nely, nelx), x-fastest
    assert float(np.mean(np.abs(field - field[::-1, :]))) < 1e-2
