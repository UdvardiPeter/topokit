# TopoKit

Open-source topology optimization for engineers.

**Status: pre-alpha.** The full numerical core runs end to end: a `Problem`
assembles mesh, physics, parametrization, objective, constraints, and an
optimizer, and a `Study` drives the loop to convergence. CAD I/O and
manufacturing constraints land next.

## Quickstart

A 60×20 cantilever, left edge fixed, downward tip load (~15 s):

```python
from topokit import (
    MMA,
    SIMP,
    Compliance,
    DensityFilter,
    LinearElasticity,
    Material,
    NearPoint,
    PlaneSlab,
    PointLoad,
    Problem,
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
chain = DensityFilter(radius=1.5) | SIMP()
problem = Problem(
    model, chain, objective=Compliance(), constraints=[Volume() <= 0.4], optimizer=MMA()
)
result = Study(problem).run()  # SIMP continuation on by default

result.design.save("cantilever.npz")
```

With the `[viz]` extra, `result.view()` renders the density field and
`result.plot_convergence()` the convergence curves.

## Goals

- Import CAD geometry, export manufacturable geometry
- Manufacturing constraints, including SLS and SLA checks
- Every layer replaceable through plugins
- Every gradient finite-difference verified in CI

## Layout

| Package | Contents |
|---|---|
| `topokit.backend` | array backend protocol, NumPy implementation, kernel registry |
| `topokit.mesh` | structured quad/hex grids, element masks, boundary faces |
| `topokit.selection` | geometric selectors for loads, supports, regions |
| `topokit.fem` | linear elasticity, loads, materials, element library |
| `topokit.solvers` | direct and AMG-preconditioned iterative solvers |
| `topokit.parametrization` | the unified chain: symmetry, filters, projection, SIMP |
| `topokit.responses` | compliance, volume, von Mises, constraint objects |
| `topokit.optimizers` | OC and clean-room MMA optimizers |
| `topokit.problem` | Problem, the Study loop, SIMP/Heaviside continuation |
| `topokit.checkpoint` | single-file `.topo` checkpoint save and resume |
| `topokit.viz` | convergence curves, density views, slices, LiveView (`[viz]` extra) |
| `topokit.jax` | JAX backend + hot kernels, selected via `use_backend("jax")` (`[jax]` extra) |
| `topokit.registry` | plugin resolution by group and name |
| `topokit.events` | typed event bus for the optimization loop |
| `topokit.fields` | validated field containers (design, element, nodal) |
| `topokit.testing` | finite-difference gradient verification |

See [ARCHITECTURE.md](https://github.com/UdvardiPeter/topokit/blob/main/ARCHITECTURE.md)
for the layer model.
