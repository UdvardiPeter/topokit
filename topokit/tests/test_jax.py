# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
import numpy as np

import topokit.jax
from topokit.backend import active_backend, default_backend, get_kernel, use_backend
from topokit.backend.conformance import ArrayBackendConformance
from topokit.fem import LinearElasticity, Material, PointLoad
from topokit.mesh import StructuredGrid
from topokit.selection import NearPoint, PlaneSlab


class TestJaxBackendConformance(ArrayBackendConformance):
    backend = topokit.jax.BACKEND


def _model() -> LinearElasticity:
    g = StructuredGrid.box(size=(6.0, 2.0, 2.0), shape=(6, 2, 2))
    left = PlaneSlab(point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), tol=1e-9)
    return LinearElasticity(
        g,
        Material(E=1.0, nu=0.3, rho=1.0),
        supports=[(left, "all")],
        loads=[PointLoad(NearPoint((6.0, 1.0, 1.0)), (0.0, -1.0, 0.0))],
    )


def test_use_backend_jax_by_name_and_restores() -> None:
    with use_backend("jax"):  # resolves via the entry point / registry
        assert active_backend().name == "jax"
        inside = get_kernel("assemble_csr_data")
    assert active_backend() is default_backend()
    outside = get_kernel("assemble_csr_data")
    assert inside is not outside  # jax kernel inside, generic fallback outside


def test_jax_assembly_matches_generic() -> None:
    model = _model()
    rng = np.random.default_rng(7)
    scale = rng.uniform(0.1, 1.0, model.mesh.n_elements)
    k0 = model.assemble(scale)  # generic/NumPy path
    with use_backend("jax"):
        k1 = model.assemble(scale)  # same model — call-time selection
    d0 = np.asarray(k0.csr_arrays()[2])
    d1 = np.asarray(k1.csr_arrays()[2])
    assert d1.dtype == np.float64
    np.testing.assert_allclose(d1, d0, rtol=1e-12, atol=1e-15)


def test_solve_through_jax_backend_matches() -> None:
    from topokit.solvers import Direct

    model = _model()
    scale = np.full(model.mesh.n_elements, 0.7)
    d = Direct()
    d.prepare(model.assemble(scale))
    u0 = d.solve(model.loads())
    with use_backend("jax"):
        d.prepare(model.assemble(scale))
        u1 = d.solve(model.loads())
    np.testing.assert_allclose(u1, u0, rtol=1e-10)
