# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
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
    from topokit.backend import use_backend

    class Named(NumpyBackend):
        @property
        def name(self) -> str:
            return "testbk"

    register_kernel("_test_exact", "testbk", lambda: "fast")
    with use_backend(Named()):
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
    from topokit.backend import NumpyBackend, active_backend, default_backend, use_backend

    class Named(NumpyBackend):
        @property
        def name(self) -> str:
            return "othernp"

    assert active_backend() is default_backend()
    other = Named()
    with use_backend(other):
        assert active_backend() is other
        with use_backend(default_backend()):  # nesting: innermost wins
            assert active_backend() is default_backend()
        assert active_backend() is other
    assert active_backend() is default_backend()


def test_use_backend_resolves_strings_via_registry() -> None:
    from topokit.backend import active_backend, default_backend, use_backend

    with use_backend("numpy"):  # registered by topokit/__init__
        assert active_backend() is default_backend()


def test_use_backend_is_thread_scoped() -> None:
    from concurrent.futures import ThreadPoolExecutor

    from topokit.backend import NumpyBackend, active_backend, default_backend, use_backend

    class Named(NumpyBackend):
        @property
        def name(self) -> str:
            return "threadnp"

    with use_backend(Named()), ThreadPoolExecutor(1) as ex:
        other = ex.submit(active_backend).result()
    assert other is default_backend()  # fresh threads start outside any context


def test_get_kernel_keys_off_active_backend() -> None:
    from topokit.backend import (
        NumpyBackend,
        get_kernel,
        register_kernel,
        use_backend,
    )

    class Named(NumpyBackend):
        @property
        def name(self) -> str:
            return "kern_test"

    register_kernel("ctx_demo", "generic", lambda: "generic")
    register_kernel("ctx_demo", "kern_test", lambda: "special")
    register_kernel("ctx_partial", "generic", lambda: "generic")
    assert get_kernel("ctx_demo")() == "generic"
    with use_backend(Named()):
        assert get_kernel("ctx_demo")() == "special"
        assert get_kernel("ctx_partial")() == "generic"  # partial coverage falls through
    assert get_kernel("ctx_demo")() == "generic"


def test_use_backend_rejects_non_backend_object() -> None:
    from topokit.backend import use_backend

    with pytest.raises(TypeError, match="use_backend expects"), use_backend(object()):  # type: ignore[arg-type]
        pass


def test_use_backend_failing_entry_point_does_not_block_other_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken plugin entry point in the ``backends`` group must not stop
    :func:`use_backend` from resolving other, healthy names.

    ``use_backend`` always consults the process-global registry, and that
    registry has already scanned the ``backends`` group by the time this
    test runs (earlier tests in this module resolve ``"numpy"`` through
    it) -- entry-point scanning happens once per group, so re-patching
    ``importlib.metadata.entry_points`` here would have no effect on it.
    To exercise the actual codepath ``use_backend`` relies on, inject one
    broken *pending* entry point directly into the global registry's
    internal state (``monkeypatch.setitem`` restores it afterwards) rather
    than building a throwaway ``Registry``.
    """
    import typing
    from importlib.metadata import EntryPoint

    from topokit.backend import active_backend, default_backend, use_backend
    from topokit.registry import registry as global_registry

    class FailingEP:
        name = "failing"
        value = "fake.mod:failing"

        def load(self) -> object:
            raise ImportError("needs the [nope] extra")

    global_registry._scan_entry_points("backends")  # ensure the group is scanned
    monkeypatch.setitem(
        global_registry._pending["backends"],
        "failing",
        typing.cast(EntryPoint, FailingEP()),
    )

    with pytest.raises(ImportError, match=r"\[nope\] extra"), use_backend("failing"):
        pass

    with use_backend("numpy"):
        assert active_backend() is default_backend()
