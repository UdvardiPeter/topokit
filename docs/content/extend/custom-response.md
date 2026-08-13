# Custom response

A response reads the solved state and returns a scalar plus its gradient. `ResponseBase` supplies the `<=`/`>=` operators that turn a response into a `Constraint`, the same way `Volume() <= 0.4` works in the tutorials. This one tracks the material centroid's x-coordinate, normalized by the domain length, and caps it below 0.45 alongside the usual volume constraint.

```python
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

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
    ResponseBase,
    Schedule,
    Solution,
    StructuredGrid,
    Study,
    Volume,
)


@dataclass(frozen=True)
class XCentroid(ResponseBase):
    """Material centroid x-coordinate, normalized by domain length."""

    name: ClassVar[str] = "x_centroid"
    field_basis: ClassVar[str] = "density"
    n_extra_adjoints: ClassVar[int] = 0

    def value(self, solution):
        mesh = solution.mesh
        w = solution.density * mesh.element_volumes
        cx = float((w * mesh.element_centroids[:, 0]).sum() / max(w.sum(), 1e-300))
        return cx / 60.0

    def grad_field(self, solution):
        mesh = solution.mesh
        v = mesh.element_volumes
        w = solution.density * v
        total = max(float(w.sum()), 1e-300)
        cx = float((w * mesh.element_centroids[:, 0]).sum() / total)
        return v * (mesh.element_centroids[:, 0] - cx) / total / 60.0
```

`field_basis="density"` tells the orchestration layer which chain output to differentiate against: `grad_field` returns a gradient with respect to the physical density, and the layer routes it back to the design variables through the chain's density pullback. A response that instead reads FE state (like `Compliance`) uses `field_basis="interpolated"`, reads `solution.displacements`, and its gradient is routed through the chain's ordinary pullback.

`XCentroid` is added as a constraint alongside `Volume`, so `MMA` (which handles more than one constraint) replaces the single-constraint `OC` from the tutorials.

```python
mesh = StructuredGrid.box(size=(60.0, 20.0), shape=(60, 20))
model = LinearElasticity(
    mesh,
    Material(E=1.0, nu=0.3, rho=1.0),
    supports=[(PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0)), "all")],
    loads=[PointLoad(NearPoint((60.0, 10.0)), force=(0.0, -1.0))],
)
chain = DensityFilter(radius=1.5) | SIMP(p=3.0)

problem = Problem(
    model,
    chain,
    objective=Compliance(),
    constraints=[Volume() <= 0.4, XCentroid() <= 0.45],
    optimizer=MMA(),
)
result = Study(problem, schedule=Schedule.single(p=3.0, max_iter=60, tol=1e-3)).run()
print(f"x_centroid {result.history['x_centroid'][-1]:.3f}")
```

Once added to `constraints` or passed as `objective`, the response appears in `result.history` under its own `name` (`"x_centroid"` here), the same place `"objective"` and every other constraint's series live.

`XCentroid` never touches `solution.displacements`, so its gradient can be checked without a physics solve: bind the chain directly, evaluate it at a design `x`, and build a `Solution` with `model=None` and placeholder displacements.

```python
from topokit.testing import assert_gradient_matches

bound = chain.bind(mesh)
x = np.full(bound.n_vars, 0.5)


def f(xx):
    ev = bound.evaluate(xx)
    sol = Solution(model=None, mesh=mesh, displacements=np.zeros((1, 1)),
                   interpolated=ev.field, density=ev.density)
    return XCentroid().value(sol)


def grad(xx):
    ev = bound.evaluate(xx)
    sol = Solution(model=None, mesh=mesh, displacements=np.zeros((1, 1)),
                   interpolated=ev.field, density=ev.density)
    return ev.pullback_density(XCentroid().grad_field(sol))


assert_gradient_matches(f, grad, x)
print("gradient verified")
```

An FE-state response needs a real solve for the same check, since its `value`/`grad_field` read `solution.displacements`.

## Next

[Custom optimizer](custom-optimizer.md) covers the other side of the loop: stepping the design variables given the values and gradients a `Problem` computes.
