# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Tests for the component registry."""

import importlib.metadata

import pytest

import topokit
from topokit.registry import Registry, RegistryError


def test_builtins_resolve_by_name() -> None:
    assert topokit.registry.get("optimizers", "mma") is topokit.MMA
    assert topokit.registry.get("solvers", "direct") is topokit.Direct
    assert topokit.registry.get("backends", "numpy").name == "numpy"


def test_register_and_get_round_trip() -> None:
    r = Registry()
    r.register("widgets", "a", object)
    assert r.get("widgets", "a") is object
    assert r.names("widgets") == ("a",)


def test_duplicate_registration_raises() -> None:
    r = Registry()
    r.register("widgets", "a", object)
    with pytest.raises(RegistryError, match="already has 'a'"):
        r.register("widgets", "a", int)


def test_unknown_name_lists_what_is_available() -> None:
    r = Registry()
    r.register("widgets", "a", object)
    with pytest.raises(RegistryError, match="no 'b' in group 'widgets'; available: a"):
        r.get("widgets", "b")


def test_unknown_group_is_empty_not_an_error() -> None:
    # groups are open: a third party inventing one does not need a code change
    r = Registry()
    assert r.names("whatever") == ()
    with pytest.raises(RegistryError, match="available: \\(none\\)"):
        r.get("whatever", "x")


def test_entry_point_is_loaded_on_demand_and_listed_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ep = importlib.metadata.EntryPoint(name="late", value="json:dumps", group="topokit.widgets")
    monkeypatch.setattr(Registry, "_entry_points", staticmethod(lambda group: (ep,)))
    r = Registry()
    assert r.names("widgets") == ("late",)  # listed without being loaded
    import json

    assert r.get("widgets", "late") is json.dumps


def test_a_failed_entry_point_load_keeps_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    ep = importlib.metadata.EntryPoint(
        name="broken", value="topokit_does_not_exist:thing", group="topokit.widgets"
    )
    monkeypatch.setattr(Registry, "_entry_points", staticmethod(lambda group: (ep,)))
    r = Registry()
    for _ in range(2):
        # the same informative error twice, not "unknown name" the second time
        with pytest.raises(ModuleNotFoundError):
            r.get("widgets", "broken")
