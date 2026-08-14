# Extend

Every pluggable seam in TopoKit is a `Protocol`: a response, an optimizer, a chain link, a physics model, a linear solver, a selector, an array backend. Implement the required methods on your own class and pass an instance where TopoKit expects one; there is no registration step. (The plugin registry exists for entry-point distribution, so a pip-installed plugin can be looked up by name; it plays no part in writing or using an extension directly.)

| To write a…      | Implement                                | Import from                |
| ---------------- | ---------------------------------------- | -------------------------- |
| response         | `Response` (`value`, `grad_field`)       | `topokit.responses`        |
| optimizer        | `Optimizer` (`setup`, `step`, `state`, `load_state`) | `topokit.optimizers` |
| chain link       | `LinkSpec.build` returning a `BoundLink` | `topokit` (top level)      |
| physics model    | `PhysicsModel`                           | `topokit.fem`              |
| linear solver    | `LinearSolver` (`prepare`, `solve`)      | `topokit.solvers`          |
| selector         | `Selector`                               | `topokit.selection`        |
| array backend    | `ArrayBackend`                           | `topokit.backend`          |

The helper vocabulary these protocols traffic in (`Solution`, `ResponseBase`, `FieldSpec`, `StepResult`, `BoundLink`, `LinkSpec`, `Mesh`) is importable from the top level too. Get a protocol wrong and it fails at `Problem(...)` construction, not partway through a solve: the error names the object and the protocol it fails to satisfy.
