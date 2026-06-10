# TopoKit

Open-source topology optimization for engineers.

**Status: pre-alpha.** The foundation layer exists (plugin registry, event bus,
field containers, gradient verification harness). Numerics land next.

## Goals

- Import CAD geometry, export manufacturable geometry
- Manufacturing constraints, including SLS and SLA checks
- Every layer replaceable through plugins
- Every gradient finite-difference verified in CI

## Layout

| Package | Contents |
|---|---|
| `topokit.backend` | array backend protocol, NumPy implementation, kernel registry |
| `topokit.registry` | plugin resolution by group and name |
| `topokit.events` | typed event bus for the optimization loop |
| `topokit.fields` | validated field containers (design, element, nodal) |
| `topokit.testing` | finite-difference gradient verification |

See [ARCHITECTURE.md](https://github.com/UdvardiPeter/topokit/blob/main/ARCHITECTURE.md)
for the layer model.
