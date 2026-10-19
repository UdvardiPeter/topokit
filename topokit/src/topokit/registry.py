# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Component lookup by ``(group, name)``.

Built-ins register themselves at package import; ``registry.get`` falls back
to the ``topokit.<group>`` entry-point namespace, which is how
``use_backend("jax")`` resolves a backend without importing it first.

Groups are open: a name registered under a group nobody has used before
creates it. The enumeration ``registry.names`` exists because the test suite
finite-difference verifies every registered chain link and response.
"""

from __future__ import annotations

import importlib.metadata
from typing import Any

from topokit.exceptions import TopoKitError


class RegistryError(TopoKitError, LookupError):
    """Unknown name, or a conflicting registration."""


class Registry:
    """Resolves components by ``(group, name)``."""

    def __init__(self) -> None:
        self._items: dict[str, dict[str, Any]] = {}

    def register(self, group: str, name: str, obj: Any) -> None:
        """Register ``obj`` under ``(group, name)``."""
        items = self._items.setdefault(group, {})
        if name in items:
            raise RegistryError(f"{group!r} already has {name!r} registered")
        items[name] = obj

    def get(self, group: str, name: str) -> Any:
        """Return the object under ``(group, name)``, loading an entry point if needed.

        A failed entry-point load raises rather than reporting the name as
        unknown, and is not cached: the next lookup retries and raises the
        same error.
        """
        items = self._items.setdefault(group, {})
        if name in items:
            return items[name]
        for ep in self._entry_points(group):
            if ep.name == name:
                obj = ep.load()
                items[name] = obj
                return obj
        available = ", ".join(self.names(group)) or "(none)"
        raise RegistryError(f"no {name!r} in group {group!r}; available: {available}")

    def names(self, group: str) -> tuple[str, ...]:
        """Sorted names in ``group``, including entry points not yet loaded."""
        declared = {ep.name for ep in self._entry_points(group)}
        return tuple(sorted(self._items.get(group, {}).keys() | declared))

    @staticmethod
    def _entry_points(group: str) -> Any:
        return importlib.metadata.entry_points(group=f"topokit.{group}")


registry = Registry()
"""Process-wide registry. Built-ins self-register on package import."""
