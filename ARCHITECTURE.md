# Architecture

TopoKit is a layered library. Each module sits behind a small protocol and can be
replaced without touching the rest. Layer boundaries are enforced in CI by
import-linter; the `boundaries` target fails on any violation.

## Layers

```
┌────────────────────────────────────────────────────────────────────┐
│  L5  CLIENTS        CAD add-ins (Fusion 360 first)                 │
│                     speak only the daemon wire protocol            │
├────────────────────────────────────────────────────────────────────┤
│  L4  SERVICE        topokit.server (local daemon, jobs, streaming) │
│                     topokit.schema (declarative problem JSON)      │
├────────────────────────────────────────────────────────────────────┤
│  L3  WORKFLOW       topokit.cadio (CAD file -> voxel domain)       │
│                     topokit.geomout (density -> STL/STEP/SDF)      │
│                     topokit.viz (PyVista live/static views)        │
├────────────────────────────────────────────────────────────────────┤
│  L2  ORCHESTRATION  topokit.problem (Problem, Study, run loop)     │
├────────────────────────────────────────────────────────────────────┤
│  L1  NUMERICS       topokit.fem        topokit.parametrization     │
│                     topokit.responses  topokit.constraints         │
│                     topokit.optimizers topokit.solvers             │
├────────────────────────────────────────────────────────────────────┤
│  L0  FOUNDATION     topokit.backend    topokit.mesh                │
│                     topokit.selection  topokit.fields              │
│                     topokit.registry   topokit.events              │
└────────────────────────────────────────────────────────────────────┘
```

A module may import from lower layers only. L0 modules are mutually independent.
L3 modules are mutually independent. Core dependencies are only NumPy and SciPy.

## Key ideas

- **Parametrization chain.** Everything between raw optimizer variables and the
  field the physics consumes is a chain of links, each with `apply` and
  `pullback` (vector-Jacobian product). Filters, projections, symmetry maps,
  and material interpolation are all links.
- **Plugin registry.** `topokit.registry` resolves components by group and name
  (`registry.get("optimizers", "mma")`). Third parties register via entry
  points in `topokit.<group>` namespaces.
- **Events.** The run loop publishes typed events (`IterationFinished`,
  `FieldSnapshot`, ...). Visualization, the daemon, and loggers are
  subscribers. Subscriber crashes are only logged.
- **Daemon for CAD clients.** CAD plugins are thin clients of a local HTTP/WS
  service.

## Test Architecture

| Tier | What | When |
|---|---|---|
| 0 | ruff, mypy strict, import-linter boundaries | every PR |
| 1 | unit tests: FEM patch and analytic cases, finite-difference gradient checks | every PR |
| 2 | the documentation's code blocks execute as written | every PR |
| 3 | benchmark regressions against published results | small on PR, full nightly |
| 4 | end-to-end CAD pipeline, performance budgets | nightly and release |

Gradient checks are Tier-1 unit tests, not a tier of their own. Every analytic
pullback is verified against a finite difference with
`topokit.testing.assert_gradient_matches`, in the 20 tests that live beside the
code they check: `test_responses.py`, `test_parametrization.py` and
`test_fem_unstructured.py`. `test_fd_harness.py` tests that harness itself.
