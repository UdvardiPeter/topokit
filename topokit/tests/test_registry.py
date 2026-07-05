# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Tests for the plugin registry."""

from collections.abc import Iterator

import pytest

from topokit.registry import GROUPS, Registry, RegistryError


def test_groups_are_the_nine_specified() -> None:
    assert GROUPS == (
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


def test_register_and_get_roundtrip() -> None:
    reg = Registry()
    sentinel = object()
    reg.register("optimizers", "mma", sentinel, source="tests")
    assert reg.get("optimizers", "mma") is sentinel


def test_unknown_group_raises_with_group_list() -> None:
    reg = Registry()
    with pytest.raises(RegistryError, match="optimizers"):
        reg.get("nonsense", "mma")


def test_unknown_name_raises_listing_available() -> None:
    reg = Registry()
    reg.register("optimizers", "oc", object(), source="tests")
    with pytest.raises(RegistryError, match="oc"):
        reg.get("optimizers", "mma")


def test_duplicate_registration_shows_both_sources() -> None:
    reg = Registry()
    reg.register("solvers", "direct", object(), source="topokit.solvers")
    with pytest.raises(RegistryError, match=r"topokit\.solvers") as exc:
        reg.register("solvers", "direct", object(), source="evil.plugin")
    assert "evil.plugin" in str(exc.value)


def test_names_sorted(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.metadata as ilm

    # A fresh Registry scans real entry points; the installed topokit.backends
    # jax entry point would collide with the seeded name.
    def no_entry_points(*, group: str) -> Iterator[object]:
        return iter([])

    monkeypatch.setattr(ilm, "entry_points", no_entry_points)
    reg = Registry()
    reg.register("backends", "numpy", object(), source="tests")
    reg.register("backends", "jax", object(), source="tests")
    assert reg.names("backends") == ("jax", "numpy")


def test_entry_points_loaded_lazily_per_group(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.metadata as ilm

    loaded = object()

    class FakeEP:
        name = "thirdparty"
        value = "fake.mod:obj"

        def load(self) -> object:
            return loaded

    def fake_entry_points(*, group: str) -> Iterator[FakeEP]:
        return iter([FakeEP()]) if group == "topokit.solvers" else iter([])

    monkeypatch.setattr(ilm, "entry_points", fake_entry_points)
    reg = Registry()
    assert reg.get("solvers", "thirdparty") is loaded
    assert "thirdparty" in reg.names("solvers")


def test_failing_entry_point_does_not_block_others_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.metadata as ilm

    healthy = object()

    class HealthyEP:
        name = "healthy"
        value = "fake.mod:healthy"

        def load(self) -> object:
            return healthy

    class FailingEP:
        name = "failing"
        value = "fake.mod:failing"
        load_calls = 0

        def load(self) -> object:
            type(self).load_calls += 1
            raise ImportError("needs the [x] extra")

    def fake_entry_points(*, group: str) -> Iterator[object]:
        return iter([HealthyEP(), FailingEP()]) if group == "topokit.exporters" else iter([])

    monkeypatch.setattr(ilm, "entry_points", fake_entry_points)
    reg = Registry()

    assert reg.get("exporters", "healthy") is healthy

    with pytest.raises(ImportError, match=r"\[x\] extra"):
        reg.get("exporters", "failing")
    assert FailingEP.load_calls == 1

    # A second lookup retries the load (and re-raises), rather than
    # degrading to "unknown name".
    with pytest.raises(ImportError, match=r"\[x\] extra"):
        reg.get("exporters", "failing")
    assert FailingEP.load_calls == 2

    assert reg.get("exporters", "healthy") is healthy


def test_names_lists_pending_entry_points_without_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.metadata as ilm

    class NeverLoadedEP:
        name = "neverloaded"
        value = "fake.mod:neverloaded"
        loaded = False

        def load(self) -> object:
            type(self).loaded = True
            return object()

    def fake_entry_points(*, group: str) -> Iterator[object]:
        return iter([NeverLoadedEP()]) if group == "topokit.importers" else iter([])

    monkeypatch.setattr(ilm, "entry_points", fake_entry_points)
    reg = Registry()

    assert "neverloaded" in reg.names("importers")
    assert NeverLoadedEP.loaded is False


def test_pending_duplicate_of_registered_name_raises_at_first_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.metadata as ilm

    class DupEP:
        name = "dup"
        value = "fake.mod:dup"
        loaded = False

        def load(self) -> object:
            type(self).loaded = True
            return object()

    def fake_entry_points(*, group: str) -> Iterator[object]:
        return iter([DupEP()]) if group == "topokit.constraints" else iter([])

    monkeypatch.setattr(ilm, "entry_points", fake_entry_points)
    reg = Registry()
    reg.register("constraints", "dup", object(), source="tests")

    with pytest.raises(RegistryError, match="dup"):
        reg.get("constraints", "dup")
    assert DupEP.loaded is False


def test_register_conflicts_with_pending_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.metadata as ilm

    class FakeEP:
        name = "dupe"
        value = "fake.module:THING"

        def load(self) -> object:
            raise AssertionError("duplicate detection must not load the entry point")

    def fake_entry_points(*, group: str) -> Iterator[FakeEP]:
        return iter([FakeEP()]) if group == "topokit.backends" else iter([])

    monkeypatch.setattr(ilm, "entry_points", fake_entry_points)
    reg = Registry()
    reg.names("backends")  # triggers the scan; "dupe" is now pending, unloaded
    with pytest.raises(RegistryError, match="dupe"):
        reg.register("backends", "dupe", object(), source="tests")
