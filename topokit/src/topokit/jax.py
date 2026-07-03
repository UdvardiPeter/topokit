# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""JAX backend and hot kernels (extra: ``[jax]``), CPU in v1.

Selection is scoped through the backend context::

    import topokit

    with topokit.use_backend("jax"):
        result = Study(problem).run()

``use_backend("jax")`` resolves this module's ``BACKEND`` through the plugin
registry (entry point ``topokit.backends``), importing it lazily. Dense array
ops and the assembly kernel run in JAX; sparse matrices and linear solves
stay on the host in v1 (the GPU work is post-v1). Importing this module sets
jax's process-global x64 flag (TopoKit is float64-only). The JAX kernels
agree with the generic ones to ~1e-12 relative, not bit-exactly.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import numpy.typing as npt

from topokit.backend import SparseMatrix, register_kernel
from topokit.backend.numpy import _ScipyCsr

try:
    import jax
    import jax.numpy as jnp
except ImportError as _exc:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "the topokit JAX backend needs the [jax] extra: pip install topokit[jax]"
    ) from _exc

import scipy.sparse

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]


class JaxBackend:
    """CPU JAX backend: dense ops on jax arrays, sparse construction on host."""

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "jax"

    def asarray(self, data: Any, dtype: npt.DTypeLike = np.float64) -> Any:
        """Convert ``data`` to a jax array."""
        return jnp.asarray(data, dtype=dtype)  # type: ignore[arg-type]

    def zeros(self, shape: tuple[int, ...], dtype: npt.DTypeLike = np.float64) -> Any:
        """Zero-filled jax array."""
        return jnp.zeros(shape, dtype=dtype)  # type: ignore[arg-type]

    def einsum(self, subscripts: str, *operands: Any) -> Any:
        """Einstein summation via ``jnp.einsum``."""
        return jnp.einsum(subscripts, *operands)

    def scatter_add(self, target: Any, indices: Any, values: Any) -> Any:
        """Functional accumulation; jax arrays are immutable, so no copy needed."""
        return jnp.asarray(target).at[jnp.asarray(indices)].add(jnp.asarray(values))

    def gather(self, source: Any, indices: Any) -> Any:
        """Take ``source[indices]``."""
        return jnp.asarray(source)[jnp.asarray(indices)]

    def coo_to_csr(self, rows: Any, cols: Any, vals: Any, shape: tuple[int, int]) -> SparseMatrix:
        """Host-side CSR; sparse algebra stays on the CPU in v1."""
        coo = scipy.sparse.coo_array(
            (np.asarray(vals), (np.asarray(rows), np.asarray(cols))), shape=shape
        )
        return _ScipyCsr(coo.tocsr())

    def csr_from_parts(
        self, data: Any, indices: Any, indptr: Any, shape: tuple[int, int]
    ) -> SparseMatrix:
        """Host-side CSR from prebuilt arrays; converts device data once."""
        return _ScipyCsr(
            scipy.sparse.csr_array(
                (np.asarray(data), np.asarray(indices), np.asarray(indptr)), shape=shape
            )
        )


BACKEND = JaxBackend()
"""Module-level instance; the ``topokit.backends`` entry point references it."""


@partial(jax.jit, static_argnames="nnz")
def _fill(scale: Any, ke: Any, pos: Any, nnz: int) -> Any:
    def body(k: Any, data: Any) -> Any:
        return data.at[pos[k]].add(ke[k] * scale)

    return jax.lax.fori_loop(0, pos.shape[0], body, jnp.zeros(nnz + 1))[:-1]


def _assemble_csr_data(scale_active: Any, ke_flat: Any, pos: Any, nnz: int) -> Any:
    """Jitted scatter fill; accepts host or device arrays."""
    return _fill(jnp.asarray(scale_active), jnp.asarray(ke_flat), jnp.asarray(pos), nnz)


register_kernel("assemble_csr_data", "jax", _assemble_csr_data)
