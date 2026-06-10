"""Tests for the array backend.

Kernel-registry state is module-global, so kernel tests use unique names
instead of fixtures.
"""

import numpy as np
import pytest

from topokit.backend import (
    ArrayBackend,
    KernelError,
    NumpyBackend,
    default_backend,
    get_kernel,
    register_kernel,
)

BK = NumpyBackend()


def test_satisfies_protocol() -> None:
    backend: ArrayBackend = BK
    assert backend.name == "numpy"


def test_asarray_defaults_to_float64() -> None:
    a = BK.asarray([1, 2, 3])
    assert a.dtype == np.float64


def test_asarray_respects_dtype() -> None:
    a = BK.asarray([1, 2, 3], dtype=np.int64)
    assert a.dtype == np.int64


def test_zeros() -> None:
    z = BK.zeros((2, 3))
    assert z.shape == (2, 3)
    assert z.dtype == np.float64
    assert not z.any()


def test_einsum() -> None:
    a = BK.asarray([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(BK.einsum("ij,j->i", a, BK.asarray([1.0, 1.0])), [3.0, 7.0])


def test_scatter_add_accumulates_duplicate_indices() -> None:
    target = BK.zeros((4,))
    out = BK.scatter_add(target, BK.asarray([0, 1, 1], dtype=np.int64), BK.asarray([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(out, [1.0, 5.0, 0.0, 0.0])


def test_scatter_add_does_not_mutate_input() -> None:
    target = BK.zeros((2,))
    BK.scatter_add(target, BK.asarray([0], dtype=np.int64), BK.asarray([1.0]))
    assert not target.any()


def test_gather() -> None:
    src = BK.asarray([10.0, 20.0, 30.0])
    np.testing.assert_allclose(BK.gather(src, BK.asarray([2, 0], dtype=np.int64)), [30.0, 10.0])


def test_default_backend_is_numpy() -> None:
    assert default_backend().name == "numpy"


def test_coo_to_csr_matvec_matches_dense() -> None:
    rows = BK.asarray([0, 0, 1, 2], dtype=np.int64)
    cols = BK.asarray([0, 2, 1, 2], dtype=np.int64)
    vals = BK.asarray([1.0, 2.0, 3.0, 4.0])
    m = BK.coo_to_csr(rows, cols, vals, shape=(3, 3))
    dense = np.zeros((3, 3))
    dense[rows, cols] = vals
    x = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(m.matvec(x), dense @ x)
    assert m.shape == (3, 3)


def test_coo_to_csr_sums_duplicate_entries() -> None:
    m = BK.coo_to_csr(
        BK.asarray([0, 0], dtype=np.int64),
        BK.asarray([0, 0], dtype=np.int64),
        BK.asarray([1.5, 2.5]),
        shape=(1, 1),
    )
    np.testing.assert_allclose(m.diagonal(), [4.0])


def test_kernel_exact_backend_match() -> None:
    register_kernel("assemble", "testbk", lambda: "fast")
    assert get_kernel("assemble", "testbk")() == "fast"


def test_kernel_falls_back_to_generic() -> None:
    register_kernel("filter", "generic", lambda: "slow")
    assert get_kernel("filter", "anything")() == "slow"


def test_kernel_missing_raises() -> None:
    with pytest.raises(KernelError, match="nope"):
        get_kernel("nope", "numpy")


def test_kernel_duplicate_raises() -> None:
    register_kernel("dup", "generic", lambda: 1)
    with pytest.raises(KernelError, match="dup"):
        register_kernel("dup", "generic", lambda: 2)


def test_numpy_backend_registered_as_builtin() -> None:
    from topokit.registry import registry

    assert registry.get("backends", "numpy") is NumpyBackend
