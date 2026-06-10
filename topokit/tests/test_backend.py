"""Tests for the array backend.

Kernel-registry state is module-global, so kernel tests use unique names
instead of fixtures.
"""

import pytest

from topokit.backend import (
    ArrayBackend,
    KernelError,
    NumpyBackend,
    default_backend,
    get_kernel,
    register_kernel,
)
from topokit.backend.conformance import ArrayBackendConformance


class TestNumpyBackend(ArrayBackendConformance):
    backend = NumpyBackend()


def test_satisfies_protocol() -> None:
    backend: ArrayBackend = NumpyBackend()
    assert backend.name == "numpy"


def test_default_backend_is_numpy() -> None:
    assert default_backend().name == "numpy"


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

    assert registry.get("backends", "numpy") is default_backend()
