# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Array-compute abstraction.

Numerics modules are written against ``ArrayBackend`` instead of importing
numpy directly, so GPU and AD backends can be added without changing them.
The protocol stays minimal: ops are added when a module needs them.

The active backend is a context selection: ``use_backend`` scopes it for a
block (strings resolve through the plugin registry's ``backends`` group);
outside any context the NumPy default applies.

Performance-critical kernels can have backend-specific implementations.
``get_kernel(name)`` resolves at call time by the active backend — first
``(name, active_backend().name)``, then the ``(name, "generic")`` fallback.

Implementations live in their own modules; ``topokit.backend.numpy`` ships
the default. The conformance suite for implementations is in
``topokit.backend.conformance``.

The plugin registry stores backend *instances* under the ``backends`` group.
Entry points in the ``topokit.backends`` namespace must reference a
module-level instance, not a class.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from topokit.registry import registry


@runtime_checkable
class SparseMatrix(Protocol):
    """CSR-semantics sparse matrix.

    Arrays are backend arrays, typed ``Any`` like the ``ArrayBackend`` ops.
    """

    @property
    def shape(self) -> tuple[int, int]:
        """Matrix dimensions."""
        ...

    def matvec(self, x: Any) -> Any:
        """Matrix-vector product."""
        ...

    def diagonal(self) -> Any:
        """Return the main diagonal as a dense vector."""
        ...

    def csr_arrays(self) -> tuple[Any, Any, Any]:
        """Return ``(indptr, indices, data)``.

        The lossless export a direct factorization needs. GPU backends may
        copy device to host here. Callers must not mutate the returned
        arrays; they may alias internal storage.
        """
        ...


class ArrayBackend(Protocol):
    """Minimal array op set the numerics need."""

    @property
    def name(self) -> str:
        """Backend identifier, e.g. ``"numpy"``."""
        ...

    def asarray(self, data: Any, dtype: npt.DTypeLike = np.float64) -> Any:
        """Convert ``data`` to a backend array.

        May return ``data`` itself when it already has the requested dtype.
        Callers must copy before mutating.
        """
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

    def csr_from_parts(
        self, data: Any, indices: Any, indptr: Any, shape: tuple[int, int]
    ) -> SparseMatrix:
        """Build a CSR matrix from prebuilt arrays without sorting or deduplication.

        The per-iteration assembly path: the pattern is fixed, only ``data``
        changes. Callers guarantee a valid canonical CSR layout.
        """
        ...


class KernelError(LookupError):
    """Unknown kernel or conflicting kernel registration."""


_KERNELS: dict[tuple[str, str], Callable[..., Any]] = {}


def register_kernel(name: str, backend_name: str, fn: Callable[..., Any]) -> None:
    """Register ``fn`` as the ``name`` kernel for ``backend_name``.

    ``"generic"`` registers the fallback used by all backends.
    Registrations are permanent: there is no unregister, and a duplicate
    name for the same backend raises ``KernelError``.
    """
    key = (name, backend_name)
    if key in _KERNELS:
        raise KernelError(f"kernel {name!r} already registered for {backend_name!r}")
    _KERNELS[key] = fn


def get_kernel(name: str) -> Callable[..., Any]:
    """Resolve ``name`` for the active backend, falling back to ``"generic"``.

    Resolve at every call site invocation — never cache the result — so
    :func:`use_backend` takes effect regardless of when objects were built.
    """
    backend_name = active_backend().name
    for key in ((name, backend_name), (name, "generic")):
        if key in _KERNELS:
            return _KERNELS[key]
    raise KernelError(f"no kernel {name!r} for backend {backend_name!r} and no generic fallback")


from topokit.backend.numpy import NumpyBackend  # noqa: E402

_DEFAULT: Final = NumpyBackend()


def default_backend() -> ArrayBackend:
    """Return the process-wide default backend."""
    return _DEFAULT


_ACTIVE: ContextVar[ArrayBackend | None] = ContextVar("topokit_backend", default=None)


def active_backend() -> ArrayBackend:
    """Return the context-selected backend, or the NumPy default outside any context."""
    backend = _ACTIVE.get()
    return backend if backend is not None else _DEFAULT


@contextmanager
def use_backend(backend: str | ArrayBackend) -> Iterator[None]:
    """Select the active backend for the dynamic extent of the block.

    Strings resolve through the plugin registry's ``backends`` group (entry
    points load lazily, so ``use_backend("jax")`` works without importing
    ``topokit.jax`` first). Scoped and nestable (innermost wins); thread- and
    async-local, so concurrent studies control their own selection. The
    context governs backend and kernel *resolution at call time*, not
    objects: where a model or chain was created is irrelevant.
    """
    resolved: ArrayBackend = (
        registry.get("backends", backend) if isinstance(backend, str) else backend
    )
    token = _ACTIVE.set(resolved)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


__all__ = [
    "ArrayBackend",
    "KernelError",
    "NumpyBackend",
    "SparseMatrix",
    "active_backend",
    "default_backend",
    "get_kernel",
    "register_kernel",
    "use_backend",
]
