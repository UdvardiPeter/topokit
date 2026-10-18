# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Conformance test suite for ``ArrayBackend`` implementations.

Subclass in a pytest suite and set ``backend``::

    from topokit.backend.conformance import ArrayBackendConformance

    class TestMyBackend(ArrayBackendConformance):
        backend = MyBackend()

Every backend, builtin or third-party, is expected to pass this suite.
It grows with the protocol; failing after a topokit upgrade means the
backend no longer satisfies the contract.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from topokit.backend import ArrayBackend


class ArrayBackendConformance:
    """Checks an ``ArrayBackend`` implementation against the protocol contract."""

    backend: ClassVar[ArrayBackend]

    def test_name_is_nonempty_string(self) -> None:
        assert isinstance(self.backend.name, str)
        assert self.backend.name

    def test_asarray_defaults_to_float64(self) -> None:
        a = self.backend.asarray([1, 2, 3])
        assert np.asarray(a).dtype == np.float64

    def test_asarray_respects_dtype(self) -> None:
        a = self.backend.asarray([1, 2, 3], dtype=np.int64)
        assert np.asarray(a).dtype == np.int64

    def test_coo_to_csr_matvec_matches_dense(self) -> None:
        rows = self.backend.asarray([0, 0, 1, 2], dtype=np.int64)
        cols = self.backend.asarray([0, 2, 1, 2], dtype=np.int64)
        vals = self.backend.asarray([1.0, 2.0, 3.0, 4.0])
        m = self.backend.coo_to_csr(rows, cols, vals, shape=(3, 3))
        dense = np.zeros((3, 3))
        dense[np.asarray(rows), np.asarray(cols)] = np.asarray(vals)
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(m.matvec(x), dense @ x)
        assert m.shape == (3, 3)

    def test_coo_to_csr_sums_duplicate_entries(self) -> None:
        m = self.backend.coo_to_csr(
            self.backend.asarray([0, 0], dtype=np.int64),
            self.backend.asarray([0, 0], dtype=np.int64),
            self.backend.asarray([1.5, 2.5]),
            shape=(1, 1),
        )
        np.testing.assert_allclose(np.asarray(m.matvec(np.ones(1))), [4.0])

    def test_csr_arrays_reconstruct_matrix(self) -> None:
        rows = self.backend.asarray([0, 0, 1, 2], dtype=np.int64)
        cols = self.backend.asarray([0, 2, 1, 2], dtype=np.int64)
        vals = self.backend.asarray([1.0, 2.0, 3.0, 4.0])
        m = self.backend.coo_to_csr(rows, cols, vals, shape=(3, 3))
        indptr, indices, data = (np.asarray(a) for a in m.csr_arrays())
        import scipy.sparse

        dense = scipy.sparse.csr_array((data, indices, indptr), shape=m.shape).toarray()
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(dense @ x, np.asarray(m.matvec(x)))

    def test_csr_from_parts_matches_dense(self) -> None:
        indptr = self.backend.asarray([0, 2, 3, 4], dtype=np.int32)
        indices = self.backend.asarray([0, 2, 1, 2], dtype=np.int32)
        data = self.backend.asarray([1.0, 2.0, 3.0, 4.0])
        m = self.backend.csr_from_parts(data, indices, indptr, shape=(3, 3))
        dense = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(np.asarray(m.matvec(x)), dense @ x)
        assert m.shape == (3, 3)
