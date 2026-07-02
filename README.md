# TopoKit

Open-source topology optimization for engineers

- Python ≥3.12 · LGPL-2.1-or-later · Nx + uv monorepo
- Dev setup: see [CONTRIBUTING.md](CONTRIBUTING.md)
- Architecture overview: see [ARCHITECTURE.md](ARCHITECTURE.md)

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
| performance pass (`assemble` hot kernel, numba `[fast]`), manufacturing constraints | next |
| CAD import/export | planned |
| Local daemon, Fusion 360 add-in | planned |

## Dev quick start

```
npm ci
uv sync --all-packages
npx nx run-many -t lint typecheck boundaries test test-fd
```
