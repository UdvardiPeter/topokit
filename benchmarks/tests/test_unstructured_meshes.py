# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from topokit_bench.problems import (
    BUILDERS,
    CASES,
    FULL_CASES,
    Builder,
    _tet10_mesh,
    _tet_mesh,
    _tri_mesh,
    make_optimizer,
)

_TET_LOCAL_EDGES: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
_REGENERATE = Path(__file__).resolve().parents[1] / "scripts" / "regenerate.py"
DATA = Path(__file__).resolve().parent / "data"

# Every frozen unstructured reference, read off the case matrices rather than
# listed, so a case added there cannot land here unchecked.
_REFERENCES: list[tuple[str, Builder, dict[str, int], str]] = [
    (f"{name}_{nelx}x{nely}_{opt}", BUILDERS[name], {"nelx": nelx, "nely": nely}, opt)
    for name, nelx, nely, opt in CASES
    if name == "cantilever_tri"
] + [
    (f"{case.key}_{opt}", case.build, dict(case.kwargs), opt)
    for case in FULL_CASES
    for opt in case.optimizers
    if "tet" in case.key
]
_REFERENCE_IDS = [stem for stem, *_ in _REFERENCES]


def test_tri_mesh_construction_and_measures() -> None:
    nelx, nely = 3, 2
    sx, sy = 6.0, 3.0
    mesh = _tri_mesh(nelx, nely, (sx, sy))
    assert mesh.element_kind == "tri3"
    assert mesh.n_elements == 2 * nelx * nely
    assert mesh.n_nodes == (nelx + 1) * (nely + 1)
    assert mesh.element_volumes.sum() == pytest.approx(sx * sy, abs=1e-12)
    faces = mesh.boundary_faces()
    assert faces.area.sum() == pytest.approx(2.0 * (sx + sy), abs=1e-12)


def test_tet_mesh_construction_and_measures() -> None:
    # Asymmetric element counts and box size per axis: a bug that mixes up
    # x/y/z (e.g. using nx1 where ny1 belongs) would corrupt connectivity or
    # the volume/surface sums here, unlike a nelx==nely==nelz cube.
    nelx, nely, nelz = 3, 2, 4
    sx, sy, sz = 3.0, 2.0, 4.0
    mesh = _tet_mesh(nelx, nely, nelz, (sx, sy, sz))
    assert mesh.element_kind == "tet4"
    assert mesh.n_elements == 6 * nelx * nely * nelz
    assert mesh.n_nodes == (nelx + 1) * (nely + 1) * (nelz + 1)
    assert mesh.element_volumes.sum() == pytest.approx(sx * sy * sz, abs=1e-12)
    faces = mesh.boundary_faces()
    surface = 2.0 * (sx * sy + sy * sz + sz * sx)
    assert faces.area.sum() == pytest.approx(surface, abs=1e-12)


def test_tet10_mesh_construction_and_measures() -> None:
    nelx, nely, nelz = 3, 2, 4
    size = (3.0, 2.0, 4.0)
    tet4 = _tet_mesh(nelx, nely, nelz, size)
    mesh = _tet10_mesh(nelx, nely, nelz, size)
    assert mesh.element_kind == "tet10"
    assert mesh.n_elements == tet4.n_elements

    # Closed-form node count: lattice nodes + one midpoint per unique tet4
    # edge, the unique-edge set derived here (not a hand constant).
    edges = {
        (int(row[a]), int(row[b])) if row[a] < row[b] else (int(row[b]), int(row[a]))
        for row in tet4.element_nodes
        for a, b in _TET_LOCAL_EDGES
    }
    assert mesh.n_nodes == tet4.n_nodes + len(edges)

    assert mesh.element_volumes.sum() == pytest.approx(size[0] * size[1] * size[2], abs=1e-12)
    faces = mesh.boundary_faces()
    surface = 2.0 * (size[0] * size[1] + size[1] * size[2] + size[2] * size[0])
    assert faces.area.sum() == pytest.approx(surface, abs=1e-12)


def _tip_face_center(kwargs: dict[str, int]) -> tuple[float, ...]:
    # what every cantilever builder means to load: mid-height of the far face
    point = [float(kwargs["nelx"]), kwargs["nely"] / 2.0]
    if "nelz" in kwargs:
        point.append(kwargs["nelz"] / 2.0)
    return tuple(point)


@pytest.mark.parametrize("build,kwargs", [r[1:3] for r in _REFERENCES], ids=_REFERENCE_IDS)
def test_frozen_case_loads_the_tip_face_center(build: Builder, kwargs: dict[str, int]) -> None:
    # NearPoint snaps to the nearest node and never complains, so a size whose
    # tip-face center is no node (an even nely on tet10, say) would silently
    # load somewhere else and the frozen reference would pin that instead. A
    # load landing on a supported node would be silently dropped by the DOF
    # condensation, which the load vector catches.
    problem = build(**kwargs, optimizer=make_optimizer("oc"))
    mesh = problem.model.mesh
    point = _tip_face_center(kwargs)
    dist = np.linalg.norm(np.asarray(mesh.nodes) - np.asarray(point), axis=1)
    assert int(np.count_nonzero(dist == 0.0)) == 1, f"no node at {point}"
    loads = np.asarray(problem.model.loads())
    assert loads.shape[1] == 1, "one load case"
    nonzero = np.flatnonzero(loads.ravel())
    assert nonzero.size == 1
    assert loads.ravel()[nonzero[0]] == -1.0


@pytest.mark.parametrize("stem,build,kwargs,opt", _REFERENCES, ids=_REFERENCE_IDS)
def test_frozen_reference_records_its_provenance(
    stem: str, build: Builder, kwargs: dict[str, int], opt: str
) -> None:
    # A frozen number is only readable against what produced it. The
    # unstructured references carry that; the structured baselines predate it
    # and pick it up on their next regeneration (see README).
    ref = np.load(DATA / f"{stem}.npz")
    mesh = build(**kwargs, optimizer=make_optimizer(opt)).model.mesh
    assert str(ref["element_kind"]) == mesh.element_kind
    assert int(ref["n_elements"]) == mesh.n_elements == ref["density"].size
    assert str(ref["optimizer"]) == opt
    # the schedule both regression suites re-run; a reference frozen under a
    # different one no longer describes the run it gates, and the coarse
    # compliance/field tolerances would not necessarily say so
    assert float(ref["penal"]) == 3.0
    assert int(ref["max_iter"]) == 100
    assert float(ref["tol"]) == 1e-3
    assert str(ref["platform"]) != ""
    assert str(ref["topokit"]) != ""


def test_regenerate_rejects_a_case_filter_that_matches_nothing() -> None:
    # The --case filter exists so the unstructured references could be frozen
    # without touching the older baselines; a mistyped filter (or the wrong
    # --only set) must not exit 0 over an unchanged tree and read as a
    # successful regeneration. The filter below matches no case, so this writes
    # nothing whatever the guard does.
    proc = subprocess.run(
        [sys.executable, str(_REGENERATE), "--only", "2d", "--case", "__no_such_case__"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "matched no reference file" in proc.stderr
    assert proc.stdout == ""
