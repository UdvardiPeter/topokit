# Checkpoints and resume

A 60×20 bridge, pinned at both bottom corners, loaded straight down at the
midpoint of the top edge. A 3D study at useful resolution runs for minutes
to hours, so this tutorial writes the run's state to disk as it goes,
interrupts it, and picks it up again exactly where it stopped.

## The bridge problem

`Box` selects every node inside an axis-aligned box between `lower` and
`upper`, padded outward by `tol`; the two boxes here pick out a small
support pad at each bottom corner.

```python
from topokit import (
    OC,
    SIMP,
    Box,
    Compliance,
    DensityFilter,
    LinearElasticity,
    Material,
    NearPoint,
    PointLoad,
    Problem,
    Schedule,
    StructuredGrid,
    Study,
    Volume,
)


def bridge() -> Problem:
    mesh = StructuredGrid.box(size=(60.0, 20.0), shape=(60, 20))
    model = LinearElasticity(
        mesh,
        Material(E=1.0, nu=0.3, rho=1.0),
        supports=[
            (Box((0.0, 0.0), (1.0, 0.0), tol=0.6), "all"),
            (Box((59.0, 0.0), (60.0, 0.0), tol=0.6), "all"),
        ],
        loads=[PointLoad(NearPoint((30.0, 20.0)), force=(0.0, -1.0))],
    )
    chain = DensityFilter(radius=1.5) | SIMP(p=3.0)
    return Problem(
        model,
        chain,
        objective=Compliance(),
        constraints=[Volume() <= 0.4],
        optimizer=OC(),
    )
```

## Checkpointing a run

`checkpoint_path` and `checkpoint_every` tell `Study` to write its state to
a `.topo` file every `checkpoint_every` iterations, and once more when
`run()` finishes. A `.topo` file is a self-describing archive: the
optimizer's internal state, the full iteration history, and a fingerprint
of the problem and schedule it was written under, all in one file. Nothing
external is needed to resume it.

`iterate()` is `run()` with the loop handed back to you: it yields one
state per iteration, so you can stop, inspect, or abandon a study
mid-flight. Breaking out of it here stands in for the thing that actually
happens, which is a crash, a full disk, or a laptop lid.

```python
study = Study(
    bridge(),
    schedule=Schedule.single(p=3.0, max_iter=60, tol=0.0),
    checkpoint_path="bridge.topo",
    checkpoint_every=10,
)
for state in study.iterate():
    if state.iteration == 20:
        break  # whatever ends a long run early
print(f"interrupted at iteration {state.iteration}, objective {state.objective:.1f}")
```

## Resuming

`Study.resume` reconstructs a `Study` from a `.topo` file and a freshly
built, equivalent `problem` argument: `bridge()` here rebuilds the same
mesh, physics, chain and optimizer from scratch. Resume checks a
fingerprint of the problem's structure against the one recorded in the
checkpoint and refuses to continue if they disagree, so resuming against a
different mesh, a different chain or different masks fails loudly instead
of mapping the stored design onto the wrong elements.

The resumed run continues under the schedule the checkpoint was written
with, from the iteration the last checkpoint recorded. Stages it already
finished stay history and are never re-run.

```python
final = Study.resume(bridge(), "bridge.topo").run()
print(f"finished at iteration {final.iterations}, compliance {final.objective:.1f}")
```

Resuming is bit-exact: the design, the optimizer's state and the objective
normalization are all restored, so an interrupted run and an uninterrupted
one reach the same answer. To run *further* than the schedule allows, build
a new study from the result instead: `Study(bridge(), schedule=longer,
x0=final.x)`.

## Next

[Visualization](visualization.md) covers rendering the design field and
convergence curves, in 2D, 3D, and live during a run.
