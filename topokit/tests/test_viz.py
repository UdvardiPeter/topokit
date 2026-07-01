"""Tier-1 tests for topokit.viz (headless: asserts on data/geometry, not pixels)."""

from topokit.viz import _backend


def test_require_matplotlib_returns_module() -> None:
    assert _backend.require_matplotlib().__name__ == "matplotlib"


def test_require_pyvista_returns_module() -> None:
    assert _backend.require_pyvista().__name__ == "pyvista"


def test_has_display_is_bool() -> None:
    assert isinstance(_backend.has_display(), bool)


def test_plot_convergence_plots_objective_and_change() -> None:
    from types import SimpleNamespace

    from topokit.viz import plot_convergence

    history = {
        "objective": [10.0, 6.0, 5.0],
        "change": [1.0, 0.3, 0.05],
        "kkt": [0.0, 0.0, 0.0],
        "stage": [0.0, 0.0, 0.0],
        "volume": [0.6, 0.5, 0.5],
    }
    result = SimpleNamespace(history=history)
    fig = plot_convergence(result)
    labels = {ln.get_label() for ax in fig.axes for ln in ax.get_lines()}
    assert "objective" in labels
    assert "change" in labels
    assert "volume" in labels  # constraint/response series
    assert "kkt" not in labels and "stage" not in labels  # bookkeeping excluded
    obj_line = next(ln for ax in fig.axes for ln in ax.get_lines() if ln.get_label() == "objective")
    assert list(obj_line.get_ydata()) == history["objective"]  # type: ignore[arg-type]
