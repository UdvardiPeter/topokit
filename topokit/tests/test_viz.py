# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Tier-1 tests for topokit.viz (headless: asserts on data/geometry, not pixels)."""

import numpy as np
import pytest

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
    from topokit.viz import VizError, view_slices

    with pytest.raises(VizError, match="2D"):
        view_slices(_design_2d(4, 3))


def test_view_slices_rejects_bad_axis_and_n() -> None:
    from topokit.viz import VizError, view_slices

    with pytest.raises(VizError, match="axis"):
        view_slices(_design_3d(6), axis="w")
    with pytest.raises(VizError, match="n must be"):
        view_slices(_design_3d(6), n=0)


def test_liveview_headless_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    import topokit.viz._live as live_mod
    from topokit.events import EventBus, FieldSnapshot
    from topokit.viz import LiveView

    monkeypatch.setattr(live_mod, "has_display", lambda: False)
    bus = EventBus()
    lv = LiveView()
    bus.subscribe(FieldSnapshot, lv)
    grid = _design_2d(4, 3).mesh
    bus.publish(FieldSnapshot(iteration=5, rho=np.linspace(0, 1, grid.n_elements), mesh=grid))
    assert lv.rendered == 0  # headless: no render happened


def test_liveview_broken_render_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    import topokit.viz._live as live_mod
    from topokit.events import EventBus, FieldSnapshot
    from topokit.viz import LiveView

    monkeypatch.setattr(live_mod, "has_display", lambda: True)
    lv = LiveView()
    monkeypatch.setattr(lv, "_draw", lambda snap: (_ for _ in ()).throw(RuntimeError("boom")))
    bus = EventBus()
    bus.subscribe(FieldSnapshot, lv)
    grid = _design_2d(4, 3).mesh
    # EventBus catches subscriber exceptions; the run must survive a broken view
    bus.publish(FieldSnapshot(iteration=5, rho=np.linspace(0, 1, grid.n_elements), mesh=grid))


def test_liveview_tracks_iteration_hud(monkeypatch: pytest.MonkeyPatch) -> None:
    import topokit.viz._live as live_mod
    from topokit.events import EventBus, IterationFinished
    from topokit.viz import LiveView

    monkeypatch.setattr(live_mod, "has_display", lambda: False)  # HUD tracks even headless
    lv = LiveView()
    bus = EventBus()
    bus.subscribe(IterationFinished, lv)
    bus.publish(
        IterationFinished(
            iteration=7, design_change=0.1, responses={"compliance": 12.5}, wall_time=0.0
        )
    )
    assert lv._hud is not None and lv._hud[0] == 7
    assert "iter 7" in lv._hud_title()
    assert "compliance=12.5" in lv._hud_title()


def test_result_sugar_delegates_to_viz() -> None:
    from matplotlib.figure import Figure

    from topokit.problem import Result

    design = _design_2d(4, 3)
    result = Result(
        design=design,
        x=design.values,
        objective=1.0,
        best_design=design,
        best_x=design.values,
        best_objective=1.0,
        history={"objective": [2.0, 1.0], "change": [0.5, 0.1]},
        iterations=2,
        stages_run=1,
        converged=True,
        reason="tol",
        timing=0.1,
        kkt=0.0,
    )
    assert isinstance(result.view(), Figure)
    assert isinstance(result.plot_convergence(), Figure)


def test_require_matplotlib_missing_gives_actionable_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "matplotlib", None)  # make `import matplotlib` raise
    with pytest.raises(ImportError, match=r"\[viz\]"):
        _backend.require_matplotlib()


def test_view_3d_offscreen_screenshot_renders() -> None:
    # Validates actual off-screen rendering. On a bare headless box (no display, no
    # OSMesa/xvfb) VTK can HARD-CRASH (segfault) here — uncatchable by try/except — so
    # skip before rendering when there is no display. The VTK geometry pipeline is
    # covered headless by test_view_3d_returns_isosurface_with_points.
    from topokit.viz import _backend

    if not _backend.has_display():
        pytest.skip("off-screen 3D rendering needs a display or virtual framebuffer (xvfb/OSMesa)")
    import pyvista as pv

    from topokit.viz import view

    plotter = view(_design_3d(6), iso=0.5, off_screen=True)
    assert isinstance(plotter, pv.Plotter)
    img = plotter.screenshot(return_img=True)
    assert img is not None and np.asarray(img).size > 0
