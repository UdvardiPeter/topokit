# Your first 2D part

A 60×20 cantilever, fixed along the left edge, loaded downward at the tip. Six objects assemble into one optimization: mesh, physics, chain, problem, study, result.

## Mesh

`StructuredGrid.box` builds a regular quad grid over a physical `size`, subdivided into `shape` elements per axis; element size falls out of that division.

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
```

## Physics

`LinearElasticity` couples the mesh to a material and boundary conditions. Supports and loads are picked geometrically rather than by node index: `PlaneSlab` selects every node on the plane through `point` with the given `normal` (here the left edge), and `NearPoint` selects the node nearest a coordinate (here the tip).

```python
model = LinearElasticity(
    mesh,
    Material(E=1.0, nu=0.3, rho=1.0),
    supports=[(PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0)), "all")],
    loads=[PointLoad(NearPoint((60.0, 10.0)), force=(0.0, -1.0))],
)
```

## Chain

The chain is the design parametrization: how raw design variables become the density field the physics sees. `DensityFilter` enforces a minimum feature size by blurring the design over a physical `radius`. `SIMP` interpolates the filtered density into a stiffness scale, penalized by `p` so intermediate densities cost more stiffness than they're worth, pushing the design toward solid-or-void.

```python
chain = DensityFilter(radius=1.5) | SIMP(p=3.0)
```

## Problem

`Problem` assembles the model, chain, objective, constraints, and optimizer into one optimization. `Volume() <= 0.4` builds a constraint from a response: `Volume()` reads the design's material fraction, and comparing it to `0.4` turns that response into a constraint capping the fraction at 40%.

```python
problem = Problem(
    model,
    chain,
    objective=Compliance(),
    constraints=[Volume() <= 0.4],
    optimizer=OC(),
)
```

## Study

`Study` drives the optimization loop to convergence. `Schedule.single(p=3.0, max_iter=60, tol=1e-3)` runs one stage at a fixed `p`. Call `Study(problem).run()` with no schedule and TopoKit runs its default SIMP/projection continuation instead: several stages that ramp `p` and the projection sharpness up in turn, which converges more reliably than fixing both from the start.

```python
result = Study(problem, schedule=Schedule.single(p=3.0, max_iter=60, tol=1e-3)).run()
print(f"compliance {result.objective:.1f} after {result.iterations} iterations")
```

## Result

`result` carries the final design and objective. `result.history` is a dict of per-iteration series, keyed by name (`"objective"`, `"change"`, one per constraint), useful for plotting convergence. `result.design.save` writes the design to a single `.npz` file.

```python
result.design.save("first-part.npz")
```

## Next

The [3D tutorial](first-3d.md) covers what changes when a model gains a third coordinate axis. [Concepts](../concepts/index.md) covers objectives, regularization, and verification in more depth.
