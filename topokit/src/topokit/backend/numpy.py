# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""NumPy implementation of the array backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import scipy.sparse

if TYPE_CHECKING:
    from topokit.backend import SparseMatrix

_F64 = npt.NDArray[np.float64]
_Arr = npt.NDArray[Any]


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

    def csr_arrays(self) -> tuple[_Arr, _Arr, _F64]:
        """Return ``(indptr, indices, data)``."""
        return self.raw.indptr, self.raw.indices, np.asarray(self.raw.data, dtype=np.float64)


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

    def csr_from_parts(
        self, data: _F64, indices: _Arr, indptr: _Arr, shape: tuple[int, int]
    ) -> SparseMatrix:
        """CSR from prebuilt arrays; no copy, no sort."""
        return _ScipyCsr(scipy.sparse.csr_array((data, indices, indptr), shape=shape))
