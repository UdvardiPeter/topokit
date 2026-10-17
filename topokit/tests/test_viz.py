# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Tier-1 tests for topokit.viz (headless: asserts on data/geometry, not pixels)."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from topokit.fields import DesignField
from topokit.mesh import StructuredGrid, UnstructuredMesh
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
    # one subplot per non-bookkeeping series (mixed scales never share an axis)
    assert [ax.get_ylabel() for ax in fig.axes] == ["objective", "change", "volume"]
    labels = {ln.get_label() for ax in fig.axes for ln in ax.get_lines()}
    assert "objective" in labels
    assert "change" in labels
    assert "volume" in labels  # constraint/response series
    assert "kkt" not in labels and "stage" not in labels  # bookkeeping excluded
    obj_line = next(ln for ax in fig.axes for ln in ax.get_lines() if ln.get_label() == "objective")
    assert list(obj_line.get_ydata()) == history["objective"]  # type: ignore[arg-type]
    assert fig.axes[-1].get_xlabel() == "iteration"


def test_plot_convergence_keys_select_subplots() -> None:
    from types import SimpleNamespace

    from topokit.viz import plot_convergence

    history = {"objective": [10.0, 6.0], "change": [1.0, 0.3]}
    fig = plot_convergence(SimpleNamespace(history=history), keys=["change"])
    assert [ax.get_ylabel() for ax in fig.axes] == ["change"]


def test_plot_convergence_marks_stage_boundaries() -> None:
    from types import SimpleNamespace

    from topokit.viz import plot_convergence

    history = {
        "objective": [10.0, 6.0, 5.0, 4.0],
        "stage": [0.0, 0.0, 1.0, 1.0],
    }
    fig = plot_convergence(SimpleNamespace(history=history))
    (ax,) = fig.axes
    vlines = [ln for ln in ax.get_lines() if ln.get_label() != "objective"]
    assert len(vlines) == 1  # one boundary, where stage 0 -> 1
    assert list(np.asarray(vlines[0].get_xdata())) == [2, 2]


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


def _tri_square() -> UnstructuredMesh:
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    elements = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return UnstructuredMesh(nodes, elements, "tri3")


def _tet_pair() -> UnstructuredMesh:
    nodes = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
    )
    elements = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    return UnstructuredMesh(nodes, elements, "tet4")


def _single_tet10() -> UnstructuredMesh:
    corners = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    midsides = np.array(
        [
            (corners[0] + corners[1]) / 2.0,
            (corners[1] + corners[2]) / 2.0,
            (corners[2] + corners[0]) / 2.0,
            (corners[0] + corners[3]) / 2.0,
            (corners[1] + corners[3]) / 2.0,
            (corners[2] + corners[3]) / 2.0,
        ]
    )
    nodes = np.vstack([corners, midsides])
    elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int64)
    return UnstructuredMesh(nodes, elements, "tet10")


def test_view_renders_2d_unstructured_as_tripcolor_figure() -> None:
    from matplotlib.figure import Figure

    from topokit.viz import view

    mesh = _tri_square()
    values = np.array([0.2, 0.9])
    fig = view(DesignField(values, mesh, name="density"))
    assert isinstance(fig, Figure)
    (ax,) = fig.axes
    assert not ax.images  # the triangulation itself, not a resampled grid image
    (triangles,) = ax.collections
    np.testing.assert_allclose(np.asarray(triangles.get_array()), values)  # per-triangle density
    np.testing.assert_allclose(
        np.asarray(triangles.get_paths()[0].vertices)[:3], mesh.nodes[mesh.element_nodes[0]]
    )
    assert triangles.get_clim() == (0.0, 1.0)  # same fixed density scale as the grid path
    assert ax.get_xlabel() == "x" and ax.get_ylabel() == "y"


def test_view_renders_3d_unstructured_as_thresholded_plotter() -> None:
    import pyvista as pv

    from topokit.viz import view
    from topokit.viz._density import _unstructured_grid

    mesh = _tet_pair()
    values = np.array([0.1, 0.9])
    field = DesignField(values, mesh, name="density")
    plotter = view(field, iso=0.5, off_screen=True)
    assert isinstance(plotter, pv.Plotter)
    assert plotter.mesh is not None or plotter.renderer.actors  # a mesh was added
    plotter.close()
    # Pin the at-or-above threshold semantics: only the 0.9 cell survives.
    assert _unstructured_grid(values, mesh).threshold(0.5, scalars="rho").n_cells == 1


def test_unstructured_grid_uses_vtk_cell_types() -> None:
    from topokit.viz._density import _unstructured_grid

    grid2 = _unstructured_grid(np.array([0.5, 0.5]), _tri_square())
    assert set(grid2.celltypes) == {5}
    grid3 = _unstructured_grid(np.array([0.5, 0.5]), _tet_pair())
    assert set(grid3.celltypes) == {10}
    assert np.allclose(grid3.cell_data["rho"], [0.5, 0.5])


def test_unstructured_grid_uses_vtk_cell_types_tet10() -> None:
    from topokit.viz._density import _unstructured_grid

    grid = _unstructured_grid(np.array([0.5]), _single_tet10())
    assert set(grid.celltypes) == {24}


def test_view_slices_rejects_unstructured() -> None:
    from topokit.viz import VizError, view_slices

    field = DesignField(np.array([0.5, 0.5]), _tet_pair(), name="density")
    with pytest.raises(VizError, match="structured grids only"):
        view_slices(field)


def test_liveview_attach_rejects_unstructured_study(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    from topokit.events import FieldSnapshot
    from topokit.fem import Material, PointLoad, UnstructuredElasticity
    from topokit.parametrization import SIMP, DensityFilter
    from topokit.problem import Problem, Study
    from topokit.responses import Compliance, Volume
    from topokit.selection import NearPoint, PlaneSlab
    from topokit.viz import LiveView, VizError

    mesh = _tri_square()
    model = UnstructuredElasticity(
        mesh,
        Material(E=1.0, nu=0.3, rho=1.0),
        supports=[(PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0)), "all")],
        loads=[PointLoad(NearPoint((1.0, 0.5)), force=(0.0, -1.0))],
    )
    problem = Problem(
        model,
        DensityFilter(radius=0.6) | SIMP(p=3.0),
        objective=Compliance(),
        constraints=[Volume() <= 0.5],
    )
    study = Study(problem)
    with pytest.raises(VizError, match="structured grids only"):
        LiveView.attach(study)
    # Nothing was subscribed: a leaked subscriber would raise inside publish, and the
    # bus logs subscriber failures instead of propagating them.
    with caplog.at_level(logging.ERROR, logger="topokit.events"):
        study.events.publish(FieldSnapshot(iteration=0, rho=np.zeros(mesh.n_elements), mesh=mesh))
    assert not caplog.records


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


def test_unstructured_cells_carry_the_mesh_geometry() -> None:
    # Area/volume comes out wrong if the VTK cell type or the node order is wrong: a
    # tet10 with its last two midsides swapped (the gmsh order) measures 1/24, not 1/6.
    from topokit.viz._density import _unstructured_grid

    for mesh in (_tri_square(), _tet_pair(), _single_tet10()):
        grid = _unstructured_grid(np.full(mesh.n_elements, 0.5), mesh)
        measured = grid.area if mesh.dim == 2 else grid.volume
        assert measured == pytest.approx(float(mesh.element_volumes.sum()))


def _foreign_field(kind: str, dim: int) -> Any:
    # A density field on a third-party Mesh whose elements the views cannot draw.
    square = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    if dim == 2:
        nodes = np.array(square)
        elements = np.arange(4)[None, :]
    else:
        nodes = np.array([[x, y, z] for z in (0.0, 1.0) for x, y in square])
        elements = np.arange(8)[None, :]
    mesh = SimpleNamespace(dim=dim, element_kind=kind, nodes=nodes, element_nodes=elements)
    return SimpleNamespace(values=np.array([0.5]), mesh=mesh)


def test_view_2d_rejects_unrenderable_element_kind() -> None:
    from topokit.viz import VizError, view

    with pytest.raises(VizError, match=r"unsupported element kind 'quad4'.*\['tri3'\]"):
        view(_foreign_field("quad4", dim=2))


def test_view_3d_rejects_unrenderable_element_kind() -> None:
    from topokit.viz import VizError, view

    with pytest.raises(VizError, match="unsupported element kind 'hex8'"):
        view(_foreign_field("hex8", dim=3), off_screen=True)


def test_view_slices_keeps_its_dim_and_axis_errors_for_unstructured() -> None:
    # The structured-only guard sits behind the argument checks, so a bad call on an
    # unstructured mesh still gets the error naming what is actually wrong with it.
    from topokit.viz import VizError, view_slices

    with pytest.raises(VizError, match="needs a 3D field"):
        view_slices(DesignField(np.array([0.5, 0.5]), _tri_square(), name="density"))
    with pytest.raises(VizError, match="axis must be one of"):
        view_slices(DesignField(np.array([0.5, 0.5]), _tet_pair(), name="density"), axis="w")


def test_liveview_draw_rejects_an_unstructured_snapshot() -> None:
    from topokit.events import FieldSnapshot
    from topokit.viz import LiveView, VizError

    mesh = _tri_square()
    with pytest.raises(VizError, match="structured grids only"):
        LiveView()._draw(FieldSnapshot(iteration=0, rho=np.zeros(2), mesh=mesh))
