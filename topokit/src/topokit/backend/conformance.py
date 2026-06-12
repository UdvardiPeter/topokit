# SPDX-License-Identifier: LGPL-2.1-or-later
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

    def test_zeros(self) -> None:
        z = np.asarray(self.backend.zeros((2, 3)))
        assert z.shape == (2, 3)
        assert z.dtype == np.float64
        assert not z.any()

    def test_zeros_respects_dtype(self) -> None:
        z = np.asarray(self.backend.zeros((2,), dtype=np.int64))
        assert z.dtype == np.int64

    def test_einsum(self) -> None:
        a = self.backend.asarray([[1.0, 2.0], [3.0, 4.0]])
        v = self.backend.asarray([1.0, 1.0])
        np.testing.assert_allclose(np.asarray(self.backend.einsum("ij,j->i", a, v)), [3.0, 7.0])

    def test_einsum_element_energy_contraction(self) -> None:
        u = self.backend.asarray([[1.0, 2.0], [3.0, 4.0]])
        k = self.backend.asarray([[2.0, 0.0], [0.0, 1.0]])
        out = np.asarray(self.backend.einsum("ei,ij,ej->e", u, k, u))
        np.testing.assert_allclose(out, [6.0, 34.0])

    def test_scatter_add_accumulates_duplicate_indices(self) -> None:
        target = self.backend.zeros((4,))
        idx = self.backend.asarray([0, 1, 1], dtype=np.int64)
        vals = self.backend.asarray([1.0, 2.0, 3.0])
        out = np.asarray(self.backend.scatter_add(target, idx, vals))
        np.testing.assert_allclose(out, [1.0, 5.0, 0.0, 0.0])

    def test_scatter_add_does_not_mutate_input(self) -> None:
        target = self.backend.zeros((2,))
        self.backend.scatter_add(
            target, self.backend.asarray([0], dtype=np.int64), self.backend.asarray([1.0])
        )
        assert not np.asarray(target).any()

    def test_gather(self) -> None:
        src = self.backend.asarray([10.0, 20.0, 30.0])
        idx = self.backend.asarray([2, 0], dtype=np.int64)
        np.testing.assert_allclose(np.asarray(self.backend.gather(src, idx)), [30.0, 10.0])

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
        np.testing.assert_allclose(m.diagonal(), [4.0])

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

    def test_diagonal(self) -> None:
        m = self.backend.coo_to_csr(
            self.backend.asarray([0, 1, 2, 0], dtype=np.int64),
            self.backend.asarray([0, 1, 2, 2], dtype=np.int64),
            self.backend.asarray([5.0, 6.0, 7.0, 9.0]),
            shape=(3, 3),
        )
        np.testing.assert_allclose(np.asarray(m.diagonal()), [5.0, 6.0, 7.0])
