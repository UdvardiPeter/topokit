# Multiple load cases

A single-case design optimizes for exactly the load it was shown, and can collapse under a load it never saw. `LinearElasticity` accepts a list of load cases instead of a single list of loads; the optimizer then designs against all of them at once.

## Mesh and physics

`loads` normally takes a flat list of `Load` objects, one case. Give it a list of lists instead and each inner list is a separate case: a 40×20 plate here, loaded down at one corner in the first case and up at the other in the second.

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
    Schedule,
    StructuredGrid,
    Study,
    Volume,
)

mesh = StructuredGrid.box(size=(40.0, 20.0), shape=(40, 20))

model = LinearElasticity(
    mesh,
    Material(E=1.0, nu=0.3, rho=1.0),
    supports=[(PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0)), "all")],
    loads=[
        [PointLoad(NearPoint((40.0, 20.0)), force=(0.0, -1.0))],
        [PointLoad(NearPoint((40.0, 0.0)), force=(0.0, 1.0))],
    ],
)
print(f"{model.n_cases} load cases")
```

`model.n_cases` reports how many cases were parsed out of `loads`.

## Chain and problem

`Compliance` sums over load cases, weighted: `Compliance(weights=(0.5, 0.5))` weighs both cases equally. Without an explicit `weights`, each case gets weight 1.0, so a single-case problem is unaffected. When `weights` is given, its length has to match `model.n_cases`; `Problem` checks this at construction, before any solve.

```python
chain = DensityFilter(radius=1.5) | SIMP(p=3.0)
problem = Problem(
    model,
    chain,
    objective=Compliance(weights=(0.5, 0.5)),
    constraints=[Volume() <= 0.4],
    optimizer=MMA(),
)
```

## Study and result

```python
result = Study(problem, schedule=Schedule.single(p=3.0, max_iter=50, tol=1e-3)).run()
print(f"compliance {result.objective:.2f}")
```

The weighted sum trades the cases off against each other: a design that is very stiff for one case and weak for the other pays for the weak one too, so the optimizer looks for a shape that does reasonably well on both.

## Next

[Symmetry, checkpoints, resume](symmetry-checkpoint.md) covers longer-running studies: resuming a study from a checkpoint and cutting search space with symmetry.
