"""Tier-1 tests for topokit.viz (headless: asserts on data/geometry, not pixels)."""

from topokit.viz import _backend


def test_require_matplotlib_returns_module() -> None:
    assert _backend.require_matplotlib().__name__ == "matplotlib"


def test_require_pyvista_returns_module() -> None:
    assert _backend.require_pyvista().__name__ == "pyvista"


def test_has_display_is_bool() -> None:
    assert isinstance(_backend.has_display(), bool)
