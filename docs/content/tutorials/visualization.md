# Visualization

Everything on this page needs the `[viz]` extra:
`pip install --pre 'topokit[viz]'`.

## Rendering a design

`result.view()` renders the final density field: a matplotlib heatmap in
2D, a PyVista iso-surface in 3D (see below). `result.plot_convergence()`
plots the objective, design change, and each constraint's response
against iteration, one subplot per series since they live on very
different scales; continuation-stage boundaries are marked with dashed
vertical lines. Both return a plain matplotlib `Figure`, not tied to
`pyplot`: display it inline in Jupyter, or save it with `fig.savefig(...)`
as below.

```python
from topokit import (
    OC,
    SIMP,
    Compliance,
    DensityFilter,
    LinearElasticity,
    Material,
    NearPoint,
    PlaneSlab,
    PointLoad,
    Problem,
    Schedule,
    StructuredGrid,
    Study,
    Volume,
)

mesh = StructuredGrid.box(size=(60.0, 20.0), shape=(60, 20))
model = LinearElasticity(
    mesh,
    Material(E=1.0, nu=0.3, rho=1.0),
    supports=[(PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0)), "all")],
    loads=[PointLoad(NearPoint((60.0, 10.0)), force=(0.0, -1.0))],
)
problem = Problem(
    model,
    DensityFilter(radius=1.5) | SIMP(p=3.0),
    objective=Compliance(),
    constraints=[Volume() <= 0.4],
    optimizer=OC(),
)
result = Study(problem, schedule=Schedule.single(p=3.0, max_iter=40, tol=1e-3)).run()

fig = result.view()
fig.savefig("design.png", dpi=150)

fig = result.plot_convergence()
fig.savefig("convergence.png", dpi=150)
```

## 3D iso-surfaces

In 3D, `result.view(iso=0.5)` returns a PyVista `Plotter` instead of a
matplotlib `Figure`: `.show()` opens an interactive window, and
`.screenshot("part.png")` renders the same scene off screen, without one:
the way to capture an image on a headless machine.

<!-- no-run -->
```python
plotter = result.view(iso=0.5)
plotter.screenshot("part.png")
```

## Slice grids

`result.view_slices(axis="z", n=3)` renders a 3D density field as a row
of `n` matplotlib cross-sections along an axis, for inspecting the
interior of a design that a single iso-surface view hides. It needs a 3D
result; this page's problem is 2D, so the snippet below is illustrative
only.

<!-- no-run -->
```python
result.view_slices(axis="z", n=3)
```

## Live monitoring

`LiveView` renders the design field as it updates, driven by the same
events `Study` publishes every iteration. Attach it before `run()`:

<!-- no-run -->
```python
from topokit.viz import LiveView

study = Study(problem, schedule=Schedule.single(p=3.0, max_iter=40, tol=1e-3))
LiveView.attach(study)
study.run()
```

It draws through matplotlib in 2D and PyVista in 3D, and is a no-op on a
machine with no display, so the same script runs unmodified on a
headless CI box or a workstation.
