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


def test_names_sorted() -> None:
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
