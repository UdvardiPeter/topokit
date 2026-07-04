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

    Entry points for a group are scanned once, on first access to that
    group; scanning only collects the entry points, it never loads them.
    Each entry point is loaded lazily, on first ``get()`` of its name -- so
    an installation missing an optional extra only pays for (and only
    fails on) the names it actually resolves. A failed load leaves the
    entry point pending: later lookups of that name retry the load and
    re-raise the same informative error, rather than silently degrading to
    "unknown name". Registrations are permanent; built-ins register at
    package import, so reloading ``topokit`` raises on the duplicate.
    """

    def __init__(self) -> None:
        self._items: dict[str, dict[str, tuple[Any, str]]] = {g: {} for g in GROUPS}
        self._pending: dict[str, dict[str, importlib.metadata.EntryPoint]] = {g: {} for g in GROUPS}
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
        """Return the object registered under ``(group, name)``.

        Loads ``name``'s entry point on first lookup, if it has not already
        been registered directly. A load failure is not cached: it leaves
        the entry point pending so the next ``get()`` of the same name
        tries again and raises the same error, instead of reporting the
        name as unknown.
        """
        self._scan_entry_points(group)
        items = self._group(group)
        if name in items:
            return items[name][0]
        pending = self._pending[group]
        if name in pending:
            ep = pending[name]
            obj = ep.load()
            del pending[name]
            self.register(group, name, obj, source=f"entry point {ep.value}")
            return obj
        available = ", ".join(self.names(group)) or "(none)"
        raise RegistryError(f"no {name!r} in group {group!r}; available: {available}")

    def names(self, group: str) -> tuple[str, ...]:
        """Return the sorted names declared in ``group``.

        Includes both registered names and pending (declared but not yet
        loaded) entry-point names; does not load anything.
        """
        self._scan_entry_points(group)
        items = self._group(group)
        pending = self._pending[group]
        return tuple(sorted(set(items) | set(pending)))

    def _group(self, group: str) -> dict[str, tuple[Any, str]]:
        if group not in self._items:
            raise RegistryError(f"unknown group {group!r}; groups: {', '.join(GROUPS)}")
        return self._items[group]

    def _scan_entry_points(self, group: str) -> None:
        if group in self._scanned or group not in self._items:
            return
        self._scanned.add(group)
        items = self._items[group]
        pending = self._pending[group]
        for ep in importlib.metadata.entry_points(group=f"{_ENTRY_POINT_PREFIX}.{group}"):
            if ep.name in items:
                raise RegistryError(
                    f"{group!r} already has {ep.name!r} registered by "
                    f"{items[ep.name][1]!r}; refusing duplicate from "
                    f"{f'entry point {ep.value}'!r}"
                )
            if ep.name in pending:
                raise RegistryError(
                    f"{group!r} already has {ep.name!r} registered by "
                    f"{f'entry point {pending[ep.name].value}'!r}; refusing duplicate "
                    f"from {f'entry point {ep.value}'!r}"
                )
            pending[ep.name] = ep


registry = Registry()
"""Process-wide default registry. Built-ins self-register on module import."""
