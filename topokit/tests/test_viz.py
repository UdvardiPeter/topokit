"""Tier-1 tests for topokit.viz (headless: asserts on data/geometry, not pixels)."""

import numpy as np

from topokit.fields import DesignField
from topokit.mesh import StructuredGrid
from topokit.viz import _backend


def _design_2d(nelx: int = 4, nely: int = 3) -> DesignField:
    grid = StructuredGrid.box(size=(float(nelx), float(nely)), shape=(nelx, nely))
    values = np.linspace(0.0, 1.0, grid.n_elements)
    return DesignField(values, grid, name="density")


def _design_3d(n: int = 4) -> DesignField:
    grid = StructuredGrid.box(size=(float(n), float(n), float(n)), shape=(n, n, n))
    values = np.linspace(0.0, 1.0, grid.n_elements)
    return DesignField(values, grid, name="density")


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


def test_view_2d_returns_figure_with_correct_orientation() -> None:
    from matplotlib.figure import Figure

    from topokit.viz import view

    design = _design_2d(4, 3)
    fig = view(design)
    assert isinstance(fig, Figure)
    img = fig.axes[0].images[0].get_array()
    expected = design.mesh.to_grid(design.values).T  # type: ignore[attr-defined]  # imshow array is (nely, nelx)
    np.testing.assert_allclose(np.asarray(img), expected)


def test_view_3d_returns_isosurface_with_points() -> None:
    import pyvista as pv

    from topokit.viz import view

    design = _design_3d(6)
    plotter = view(design, iso=0.5, off_screen=True)
    assert isinstance(plotter, pv.Plotter)
    assert any(getattr(m, "n_points", 0) > 0 for m in plotter.meshes)


def test_view_slices_3d_returns_n_subplots() -> None:
    from matplotlib.figure import Figure

    from topokit.viz import view_slices

    fig = view_slices(_design_3d(6), axis="z", n=3)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3


def test_view_slices_2d_raises() -> None:
    import pytest

    from topokit.viz import VizError, view_slices

    with pytest.raises(VizError, match="2D"):
        view_slices(_design_2d(4, 3))
