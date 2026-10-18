# SPDX-License-Identifier: MIT
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


class ScipyCsr:
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

    def coo_to_csr(
        self, rows: _Arr, cols: _Arr, vals: _Arr, shape: tuple[int, int]
    ) -> SparseMatrix:
        """Build CSR from COO triplets. scipy sums duplicates on conversion."""
        coo = scipy.sparse.coo_array((vals, (rows, cols)), shape=shape)
        return ScipyCsr(coo.tocsr())

    def csr_from_parts(
        self, data: _F64, indices: _Arr, indptr: _Arr, shape: tuple[int, int]
    ) -> SparseMatrix:
        """CSR from prebuilt arrays; no copy, no sort."""
        return ScipyCsr(scipy.sparse.csr_array((data, indices, indptr), shape=shape))
