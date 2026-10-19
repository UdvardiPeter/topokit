# API reference

Every public name, grouped by what you need it for. The headings give the
module a name is defined in; the vocabulary a script uses is also importable
straight from `topokit` (`from topokit import StructuredGrid, Problem, ...`),
while the module-level helpers -- `topokit.checkpoint.read_topo`,
`topokit.viz.view` -- are reached by their module path.

## Setting up a problem

The objects a run is assembled from: a mesh, the selectors that say *where*
on it, a physics model, the parametrization chain between design variables
and the field the physics consumes, the responses to optimize and bound, and
the optimizer and solver that drive it.

### topokit.mesh

::: topokit.mesh

### topokit.selection

::: topokit.selection

### topokit.fem

::: topokit.fem

### topokit.parametrization

::: topokit.parametrization

### topokit.responses

::: topokit.responses

### topokit.optimizers

::: topokit.optimizers

### topokit.solvers

::: topokit.solvers

### topokit.problem

::: topokit.problem

## Results, fields and events

What a run hands back, what it emits while running, and how to save and
resume it.

### topokit.fields

::: topokit.fields

### topokit.events

::: topokit.events

### topokit.checkpoint

::: topokit.checkpoint

### topokit.viz

Needs the `[viz]` extra.

::: topokit.viz

## Extending TopoKit

The authoring surface: how a third-party component registers itself, the
array backend the numerics are written against, and the finite-difference
check every gradient is held to. The [Extend](../extend/index.md) guides
walk through each one.

### topokit.registry

::: topokit.registry

### topokit.backend

::: topokit.backend

### topokit.backend.numpy

::: topokit.backend.numpy

### topokit.backend.conformance

::: topokit.backend.conformance

### topokit.testing

::: topokit.testing

### topokit.jax

Needs the `[jax]` extra.

::: topokit.jax

## Errors

Every error type carries its module's name and subclasses both `TopoKitError`
and a builtin (`MeshError` is a `ValueError`, `SolverError` a `RuntimeError`),
so `except TopoKitError` catches a whole run and `except ValueError` keeps
working. They are documented with their own modules above.

### topokit.exceptions

::: topokit.exceptions
