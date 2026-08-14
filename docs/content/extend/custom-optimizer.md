# Custom optimizer

An optimizer is a pure stepper over the design variables: given where the design is now and the objective/constraint values and gradients there, it returns where to go next. It never touches the mesh, physics, or chain; those live in `Problem` and `Study`. The protocol has four methods:

- `setup(n_vars, lower, upper)`: called once before stepping, with the variable count and the box bounds.
- `step(x, f0, df0, g, dg)`: called once per iteration; returns a `StepResult`.
- `state()`: returns serializable state for checkpointing.
- `load_state(state)`: restores from `state()`.

`ProjectedGradient` below is steepest descent with a move limit, handling its one volume constraint by bisecting a Lagrange multiplier until the constraint is satisfied at the clipped step.

```python
import numpy as np

from topokit import (
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
    StepResult,
    StructuredGrid,
    Study,
    Volume,
)


class ProjectedGradient:
    """Steepest descent with a move limit; one constraint via multiplier bisection."""

    def __init__(self, lr: float = 0.1) -> None:
        self.lr = lr
        self._lower = None
        self._upper = None

    def setup(self, n_vars, lower, upper):
        self._lower, self._upper = lower, upper

    def step(self, x, f0, df0, g, dg):
        lo = np.maximum(self._lower, x - self.lr)
        hi = np.minimum(self._upper, x + self.lr)

        def candidate(lam):
            return np.clip(x - self.lr * (df0 + lam * dg[0]), lo, hi)

        l1, l2 = 0.0, 1e6
        for _ in range(100):
            lm = 0.5 * (l1 + l2)
            if g[0] + dg[0] @ (candidate(lm) - x) > 0:
                l1 = lm
            else:
                l2 = lm
        xn = candidate(0.5 * (l1 + l2))
        return StepResult(x_next=xn, change=float(np.abs(xn - x).max()))

    def state(self):
        return {}

    def load_state(self, state):
        return None


mesh = StructuredGrid.box(size=(60.0, 20.0), shape=(60, 20))
model = LinearElasticity(
    mesh,
    Material(E=1.0, nu=0.3, rho=1.0),
    supports=[(PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0)), "all")],
    loads=[PointLoad(NearPoint((60.0, 10.0)), force=(0.0, -1.0))],
)
problem = Problem(
    model,
    DensityFilter(radius=1.5) | SIMP(p=3.0),
    objective=Compliance(),
    constraints=[Volume() <= 0.4],
    optimizer=ProjectedGradient(lr=0.1),
)
result = Study(problem, schedule=Schedule.single(p=3.0, max_iter=40, tol=1e-4)).run()
print(f"compliance {result.history['objective'][0]:.1f} -> {result.objective:.1f}")
```

`f0`/`df0` arrive normalized by the magnitude of the stage's initial objective, so a well-scaled optimizer sees values of order one regardless of the problem's raw units. `g`/`dg` arrive unscaled, in `g <= 0` form: satisfied means non-positive, and `dg[i]` is the gradient of `g[i]` with respect to the design variables.

`state`/`load_state` exist so a checkpoint can carry an optimizer's internal memory (MMA's move limits, OC's Lagrange multiplier history) across a resume; a stateless method like this one just returns `{}` and ignores whatever it's handed back.

Naming a hyperparameter `step` would shadow the `step` method with a plain attribute, and `Problem` rejects that at construction rather than letting it fail the first time the loop tries to call it:

```
ProblemError: optimizer 'BadOptimizer' does not satisfy the Optimizer protocol (setup/step/state/load_state)
```

## Next

[Custom link](custom-link.md) covers extending the parametrization chain, and [custom physics](custom-physics.md) covers a different `PhysicsModel`.
