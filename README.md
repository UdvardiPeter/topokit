# TopoKit

Open-source topology optimization for engineers

- Python ≥3.12 · LGPL-2.1-or-later · Nx + uv monorepo
- Dev setup: see [CONTRIBUTING.md](CONTRIBUTING.md)
- Architecture overview: see [ARCHITECTURE.md](ARCHITECTURE.md)

## Status

Pre-alpha. Foundation layer only.

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
| `topokit.optimizers` OC (optimality criteria) | done |
| `topokit.optimizers` OC + clean-room MMA | done |
| Problem/Study orchestration | next |
| CAD import/export, manufacturing constraints | planned |
| Local daemon, Fusion 360 add-in | planned |

## Dev quick start

```
npm ci
uv sync --all-packages
npx nx run-many -t lint typecheck boundaries test test-fd
```
