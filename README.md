# TopoKit

Open-source topology optimization for engineers

[![Docs](https://app.readthedocs.org/projects/topokit/badge/?version=latest)](https://topokit.readthedocs.io/en/latest/)

- Python ≥3.12 · LGPL-2.1-or-later · Nx + uv monorepo
- Docs: [topokit.readthedocs.io](https://topokit.readthedocs.io/)
- Dev setup: see [CONTRIBUTING.md](CONTRIBUTING.md)
- Architecture overview: see [ARCHITECTURE.md](ARCHITECTURE.md)

## Quickstart

A 60×20 cantilever, left edge fixed, downward tip load, runs in about 15 s.
Install with `pip install --pre topokit` (pre-alpha, published as a dev
release):

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
print(f"compliance {result.objective:.1f} after {result.iterations} iterations")
```

With the `[viz]` extra, `result.view()` renders the density field and
`result.plot_convergence()` the objective/constraint/design-change curves.
The nightly suite executes this snippet; if it drifts from the code, CI fails.

## Status

Pre-alpha. The numerical core runs end to end (2D and 3D)

| Module | State |
|---|---|
| `topokit.backend` array backend, kernel registry | done |
| `topokit.mesh` structured grids, masks, boundary faces | done |
| `topokit.selection` geometric selectors | done |
| `topokit.registry` plugin registry | done |
| `topokit.events` event bus | done |
| `topokit.fields` field containers | done |
| `topokit.testing` FD gradient verification | done |
| `topokit.fem` linear elasticity, loads, materials | done |
| `topokit.solvers` direct + AMG-preconditioned CG | done |
| `topokit.parametrization` the unified SIMP chain | done |
| `topokit.responses` compliance, volume, constraints | done |
| `topokit.optimizers` OC + clean-room MMA | done |
| `topokit.problem` Problem + Study, continuation, checkpoint/restart | done |
| `topokit.checkpoint` single-file `.topo` save/resume | done |
| `benchmarks` 2D reference regression — MBB + cantilever vs 88-line (Phase 1 gate) | done |
| `topokit.parametrization` radial density filter | done |
| `benchmarks` 3D regression — cantilever + Michell + robustness sweep (nightly) | done |
| `benchmarks` perf scaling-study baseline + `nightly.yml` cron | done |
| `topokit.viz` convergence curves, density views, slices, LiveView (`[viz]` extra) | done |
| `topokit.jax` JAX backend + assembly kernels via `use_backend("jax")` (`[jax]` extra) | done |
| AMG near-nullspace + perf budgets/gates, manufacturing constraints | next |
| CAD import/export | planned |
| Local daemon, Fusion 360 add-in | planned |

## Dev quick start

```
npm ci
uv sync --all-packages
npx nx run-many -t lint typecheck boundaries test test-fd
```
