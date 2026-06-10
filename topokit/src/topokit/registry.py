# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Entry-point based plugin registry.

Components resolve by group and name: ``registry.get("optimizers", "mma")``.
Third-party packages register through entry points in the ``topokit.<group>``
namespaces.
"""

from __future__ import annotations

import importlib.metadata
from typing import Any, Final

GROUPS: Final = (
    "backends",
    "physics",
    "chain_links",
    "responses",
    "constraints",
    "optimizers",
    "solvers",
    "importers",
    "exporters",
)
_ENTRY_POINT_PREFIX: Final = "topokit"


class RegistryError(LookupError):
    """Unknown group or name, or a conflicting registration."""


class Registry:
    """Resolves components by ``(group, name)``.

    Entry points for a group are scanned once, on first access to that group.
    Registrations are permanent; built-ins register at package import, so
    reloading ``topokit`` raises on the duplicate.
    """

    def __init__(self) -> None:
        self._items: dict[str, dict[str, tuple[Any, str]]] = {g: {} for g in GROUPS}
        self._scanned: set[str] = set()

    def register(self, group: str, name: str, obj: Any, *, source: str) -> None:
        """Register ``obj`` under ``(group, name)``.

        ``source`` is shown in conflict error messages.
        """
        items = self._group(group)
        if name in items:
            raise RegistryError(
                f"{group!r} already has {name!r} registered by {items[name][1]!r}; "
                f"refusing duplicate from {source!r}"
            )
        items[name] = (obj, source)

    def get(self, group: str, name: str) -> Any:
        """Return the object registered under ``(group, name)``."""
        self._scan_entry_points(group)
        items = self._group(group)
        if name not in items:
            available = ", ".join(self.names(group)) or "(none)"
            raise RegistryError(f"no {name!r} in group {group!r}; available: {available}")
        return items[name][0]

    def names(self, group: str) -> tuple[str, ...]:
        """Return the sorted names registered in ``group``."""
        self._scan_entry_points(group)
        return tuple(sorted(self._group(group)))

    def _group(self, group: str) -> dict[str, tuple[Any, str]]:
        if group not in self._items:
            raise RegistryError(f"unknown group {group!r}; groups: {', '.join(GROUPS)}")
        return self._items[group]

    def _scan_entry_points(self, group: str) -> None:
        if group in self._scanned or group not in self._items:
            return
        self._scanned.add(group)
        for ep in importlib.metadata.entry_points(group=f"{_ENTRY_POINT_PREFIX}.{group}"):
            self.register(group, ep.name, ep.load(), source=f"entry point {ep.value}")


registry = Registry()
"""Process-wide default registry. Built-ins self-register on module import."""
