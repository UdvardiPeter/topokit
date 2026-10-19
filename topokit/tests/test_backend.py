# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Tests for the array backend.

Kernel-registry state is module-global, so kernel tests use unique names
instead of fixtures.
"""

import pytest

from tests.conftest import named_backend
from topokit.backend import (
    ArrayBackend,
    KernelError,
    NumpyBackend,
    active_backend,
    default_backend,
    get_kernel,
    register_kernel,
    use_backend,
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
    register_kernel("_test_exact", "testbk", lambda: "fast")
    with use_backend(named_backend("testbk")):
        assert get_kernel("_test_exact")() == "fast"


def test_kernel_falls_back_to_generic() -> None:
    register_kernel("_test_fallback", "generic", lambda: "slow")
    assert get_kernel("_test_fallback")() == "slow"


def test_kernel_missing_raises() -> None:
    with pytest.raises(KernelError, match="_test_missing"):
        get_kernel("_test_missing")


def test_kernel_duplicate_raises() -> None:
    register_kernel("_test_dup", "generic", lambda: 1)
    with pytest.raises(KernelError, match="_test_dup"):
        register_kernel("_test_dup", "generic", lambda: 2)


def test_numpy_backend_registered_as_builtin() -> None:
    from topokit.registry import registry

    assert registry.get("backends", "numpy") is default_backend()


def test_use_backend_overrides_and_restores() -> None:
    assert active_backend() is default_backend()
    other = named_backend("othernp")
    with use_backend(other):
        assert active_backend() is other
        with use_backend(default_backend()):  # nesting: innermost wins
            assert active_backend() is default_backend()
        assert active_backend() is other
    assert active_backend() is default_backend()


def test_use_backend_resolves_strings_via_registry() -> None:
    with use_backend("numpy"):  # registered by topokit/__init__
        assert active_backend() is default_backend()


def test_use_backend_is_thread_scoped() -> None:
    from concurrent.futures import ThreadPoolExecutor

    with use_backend(named_backend("threadnp")), ThreadPoolExecutor(1) as ex:
        other = ex.submit(active_backend).result()
    assert other is default_backend()  # fresh threads start outside any context


def test_get_kernel_keys_off_active_backend() -> None:
    register_kernel("ctx_demo", "generic", lambda: "generic")
    register_kernel("ctx_demo", "kern_test", lambda: "special")
    register_kernel("ctx_partial", "generic", lambda: "generic")
    assert get_kernel("ctx_demo")() == "generic"
    with use_backend(named_backend("kern_test")):
        assert get_kernel("ctx_demo")() == "special"
        assert get_kernel("ctx_partial")() == "generic"  # partial coverage falls through
    assert get_kernel("ctx_demo")() == "generic"


def test_use_backend_rejects_non_backend_object() -> None:
    with pytest.raises(TypeError, match="use_backend expects"), use_backend(object()):  # type: ignore[arg-type]
        pass


def test_use_backend_failing_entry_point_does_not_block_other_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a broken plugin in the backends group must not stop a healthy name resolving
    from importlib.metadata import EntryPoint

    from topokit.registry import Registry

    broken = EntryPoint(name="failing", value="topokit_nope:BACKEND", group="topokit.backends")
    monkeypatch.setattr(Registry, "_entry_points", staticmethod(lambda group: (broken,)))

    with pytest.raises(ModuleNotFoundError), use_backend("failing"):
        pass

    with use_backend("numpy"):
        assert active_backend() is default_backend()
