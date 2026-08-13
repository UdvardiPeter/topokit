# Your first 3D part

A 16x8x8 block, fixed on one face, loaded at the opposite corner. Same six objects as the 2D tutorial; what changes is the coordinate dimension and, under the hood, the solver.

## Mesh

`StructuredGrid.box` takes a third axis the same way it takes the first two: `size` and `shape` just grow a component.

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

mesh = StructuredGrid.box(size=(16.0, 8.0, 8.0), shape=(16, 8, 8))
```

## Physics

`PlaneSlab` and `NearPoint` take three-component `point`/`normal` tuples in 3D, and `PointLoad.force` takes a three-component vector. Here the plane through the origin with normal `(1, 0, 0)` is the whole face at x=0, and the load lands at the corner nearest `(16, 4, 4)`.

```python
model = LinearElasticity(
    mesh,
    Material(E=1.0, nu=0.3, rho=1.0),
    supports=[(PlaneSlab(point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0)), "all")],
    loads=[PointLoad(NearPoint((16.0, 4.0, 4.0)), force=(0.0, 0.0, -1.0))],
)
```

## Chain and problem

The chain and problem assembly are dimension-agnostic; nothing here changes from 2D.

```python
chain = DensityFilter(radius=1.5) | SIMP(p=3.0)
problem = Problem(
    model,
    chain,
    objective=Compliance(),
    constraints=[Volume() <= 0.3],
    optimizer=OC(),
)
```

`Problem` takes a `solver` argument, defaulting to `"auto"`. `auto` picks a solver by system size and dimension: a direct factorization for small systems, and AMG-preconditioned CG for large 3D ones, where a factorization's fill-in gets expensive. The AMG path needs `pyamg`, installed via `pip install topokit[fast]`; without it, `auto` falls back to the direct solver and warns. This tutorial's mesh is small enough to factorize directly either way.

## Study and result

```python
result = Study(problem, schedule=Schedule.single(p=3.0, max_iter=40, tol=1e-3)).run()
print(f"compliance {result.objective:.1f}, volume {result.history['volume'][-1]:.3f}")
```

16x8x8 elements is sized to run in a tutorial, not to produce a part worth printing; production runs use finer meshes at correspondingly higher cost. For a full-size result, see the Gallery's [3D cantilever](../gallery/cantilever-3d.md).

## Inspecting a 3D design

A 2D design is one image; a 3D density field needs slicing to see the interior. `result.view_slices` renders a row of cross-sections along an axis. It needs the `[viz]` extra (`pip install topokit[viz]`).

<!-- no-run -->
```python
result.view_slices(axis="z", n=3)
```

## Next

The [multiple load cases](multi-load.md) tutorial covers designs that have to survive more than one loading scenario.
