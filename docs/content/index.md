# TopoKit

Open-source topology optimization for engineers.

**Status: pre-alpha.** The full numerical core runs end to end: a `Problem`
assembles mesh, physics, parametrization, objective, constraints, and an
optimizer, and a `Study` drives the loop to convergence. CAD I/O and
manufacturing constraints land next.

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
| `topokit.problem` | Problem assembly and the Study optimization loop |
| `topokit.registry` | plugin resolution by group and name |
| `topokit.events` | typed event bus for the optimization loop |
| `topokit.fields` | validated field containers (design, element, nodal) |
| `topokit.testing` | finite-difference gradient verification |

See [ARCHITECTURE.md](https://github.com/UdvardiPeter/topokit/blob/main/ARCHITECTURE.md)
for the layer model.
