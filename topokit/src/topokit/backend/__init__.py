# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Array-compute abstraction.

Numerics modules are written against ``ArrayBackend`` instead of importing
numpy directly, so GPU and AD backends can be added without changing them.
The protocol stays minimal: ops are added when a module needs them.

Performance-critical kernels can have backend-specific implementations.
``get_kernel`` resolves by ``(name, backend_name)`` and falls back to the
``"generic"`` implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import scipy.sparse

_F64 = npt.NDArray[np.float64]
_Arr = npt.NDArray[Any]


@runtime_checkable
class SparseMatrix(Protocol):
    """CSR-semantics sparse matrix."""

    @property
    def shape(self) -> tuple[int, int]:
        """Matrix dimensions."""
        ...

    def matvec(self, x: _F64) -> _F64:
        """Matrix-vector product."""
        ...

    def diagonal(self) -> _F64:
        """Return the main diagonal as a dense vector."""
        ...


class ArrayBackend(Protocol):
    """Minimal array op set the numerics need."""

    @property
    def name(self) -> str:
        """Backend identifier, e.g. ``"numpy"``."""
        ...

    def asarray(self, data: Any, dtype: npt.DTypeLike = np.float64) -> Any:
        """Convert ``data`` to a backend array."""
        ...

    def zeros(self, shape: tuple[int, ...], dtype: npt.DTypeLike = np.float64) -> Any:
        """Zero-filled array."""
        ...

    def einsum(self, subscripts: str, *operands: Any) -> Any:
        """Einstein summation."""
        ...

    def scatter_add(self, target: Any, indices: Any, values: Any) -> Any:
        """Return a copy of ``target`` with ``values`` accumulated at ``indices``."""
        ...

    def gather(self, source: Any, indices: Any) -> Any:
        """Take ``source[indices]``."""
        ...

    def coo_to_csr(self, rows: Any, cols: Any, vals: Any, shape: tuple[int, int]) -> SparseMatrix:
        """Build a CSR matrix from COO triplets. Duplicate entries are summed."""
        ...


class _ScipyCsr:
    """``SparseMatrix`` over ``scipy.sparse.csr_array``.

    The underlying scipy array is exposed as ``raw`` for solver use.
    """

    def __init__(self, raw: scipy.sparse.csr_array) -> None:
        self.raw = raw

    @property
    def shape(self) -> tuple[int, int]:
        """Matrix dimensions."""
        return self.raw.shape

    def matvec(self, x: _F64) -> _F64:
        """Matrix-vector product."""
        return np.asarray(self.raw @ x, dtype=np.float64)

    def diagonal(self) -> _F64:
        """Return the main diagonal as a dense vector."""
        return np.asarray(self.raw.diagonal(), dtype=np.float64)


class NumpyBackend:
    """Default CPU backend."""

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "numpy"

    def asarray(self, data: Any, dtype: npt.DTypeLike = np.float64) -> _Arr:
        """Convert ``data`` to an ndarray."""
        return np.asarray(data, dtype=dtype)

    def zeros(self, shape: tuple[int, ...], dtype: npt.DTypeLike = np.float64) -> _Arr:
        """Zero-filled ndarray."""
        return np.zeros(shape, dtype=dtype)

    def einsum(self, subscripts: str, *operands: Any) -> _Arr:
        """Einstein summation via ``np.einsum``."""
        return np.einsum(subscripts, *operands)  # type: ignore[no-any-return]

    def scatter_add(self, target: _Arr, indices: _Arr, values: _Arr) -> _Arr:
        """Unbuffered accumulation via ``np.add.at`` on a copy."""
        out = target.copy()
        np.add.at(out, indices, values)
        return out

    def gather(self, source: _Arr, indices: _Arr) -> _Arr:
        """Take ``source[indices]``."""
        return source[indices]

    def coo_to_csr(
        self, rows: _Arr, cols: _Arr, vals: _Arr, shape: tuple[int, int]
    ) -> SparseMatrix:
        """Build CSR from COO triplets. scipy sums duplicates on conversion."""
        coo = scipy.sparse.coo_array((vals, (rows, cols)), shape=shape)
        return _ScipyCsr(coo.tocsr())


_DEFAULT: Final = NumpyBackend()


def default_backend() -> ArrayBackend:
    """Return the process-wide default backend."""
    return _DEFAULT


class KernelError(LookupError):
    """Unknown kernel or conflicting kernel registration."""


_KERNELS: dict[tuple[str, str], Callable[..., Any]] = {}


def register_kernel(name: str, backend_name: str, fn: Callable[..., Any]) -> None:
    """Register ``fn`` as the ``name`` kernel for ``backend_name``.

    ``"generic"`` registers the fallback used by all backends.
    """
    key = (name, backend_name)
    if key in _KERNELS:
        raise KernelError(f"kernel {name!r} already registered for {backend_name!r}")
    _KERNELS[key] = fn


def get_kernel(name: str, backend_name: str) -> Callable[..., Any]:
    """Resolve the ``name`` kernel for ``backend_name``, falling back to ``"generic"``."""
    for key in ((name, backend_name), (name, "generic")):
        if key in _KERNELS:
            return _KERNELS[key]
    raise KernelError(f"no kernel {name!r} for backend {backend_name!r} and no generic fallback")
