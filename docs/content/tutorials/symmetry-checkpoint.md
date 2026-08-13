# Symmetry, checkpoints, resume

A 60x20 bridge, pinned at both bottom corners, loaded straight down at the
midpoint of the top edge. The geometry, load, and supports are all
mirror-symmetric about the vertical centerline, so the optimal design is
too. This tutorial enforces that symmetry, checkpoints a run partway
through, and resumes it later with a longer schedule.

## The bridge problem

`Box` selects every node inside an axis-aligned box between `lower` and
`upper`, padded outward by `tol`; the two boxes here pick out a small
support pad at each bottom corner.

`SymmetryMap(planes=("x",))` folds the design across the plane normal to
`x` (the vertical centerline): every pair of mirrored elements is driven
by one shared design variable, so the mirror is exact by construction
rather than approximate. It halves the design space: 1200 elements
reduce to 600 design variables here, so each iteration is also cheaper.
`SymmetryMap` reduces the variable count every later link in the chain
works with, so it has to run first: the binder rejects a reduced-input
link anywhere else in the chain.

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
    SymmetryMap,
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
    chain = SymmetryMap(planes=("x",)) | DensityFilter(radius=1.5) | SIMP(p=3.0)
    return Problem(
        model,
        chain,
        objective=Compliance(),
        constraints=[Volume() <= 0.4],
        optimizer=OC(),
    )
```

## Checkpointing a run

`checkpoint_path` and `checkpoint_every` tell `Study` to write its state
to a `.topo` file every `checkpoint_every` iterations, and once more at
the end of `run()`. A `.topo` file is a self-describing archive: the
optimizer's internal state, the full iteration history, and a fingerprint
of the problem and schedule it was written under, all in one file;
nothing external is needed to resume it.

The schedule below caps out at 20 iterations before the run stops, well
short of convergence (`tol=0.0` never triggers the early-stop check, so
`partial.converged` is `False`).

```python
first = Study(
    bridge(),
    schedule=Schedule.single(p=3.0, max_iter=20, tol=0.0),
    checkpoint_path="bridge.topo",
    checkpoint_every=10,
)
partial = first.run()
print(f"stopped at iteration {partial.iterations}, converged={partial.converged}")
```

## Resuming

`Study.resume` reconstructs a `Study` from a `.topo` file and a freshly
built, equivalent `problem` argument: `bridge()` here rebuilds the same
mesh, physics, chain, and optimizer from scratch. Resume checks a
fingerprint of the problem's structure against the one recorded in the
checkpoint and refuses to continue if they don't match, so resuming
against a different problem fails loudly instead of silently continuing
an incompatible run.

`schedule` is optional; when given, it replaces the run's schedule going
forward, and may extend `max_iter` beyond the checkpointed one. Stages
the checkpoint already completed stay as history and are never re-run.
Here the same single stage grows from 20 to 60 iterations, so the
resumed run picks up mid-stage where the checkpoint left off and
continues to 60.

```python
resumed = Study.resume(
    bridge(),
    "bridge.topo",
    schedule=Schedule.single(p=3.0, max_iter=60, tol=0.0),
)
final = resumed.run()
print(f"finished at iteration {final.iterations}, compliance {final.objective:.1f}")
```

## Next

[Visualization](visualization.md) covers rendering the design field and
convergence curves, in 2D, 3D, and live during a run.
