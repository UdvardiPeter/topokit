# Cantilever

A 120x40 cantilever: the left edge is fully fixed, a downward point load at the mid-height of the right edge. Volume fraction 0.4.

![Cantilever](img/cantilever.png)

Needs the `[viz]` extra (`pip install --pre 'topokit[viz]'`) for `result.view()`. This script runs several minutes at full size; it is not part of the nightly docs harness, but the same physics runs there through the benchmark suite.

<!-- no-run -->
```python
from topokit import (
    OC,
    SIMP,
    Compliance,
    LinearElasticity,
    Material,
    NearPoint,
    PlaneSlab,
    PointLoad,
    Problem,
    RadialDensityFilter,
    Schedule,
    StructuredGrid,
    Study,
    Volume,
)

nelx, nely = 120, 40
mesh = StructuredGrid.box(size=(float(nelx), float(nely)), shape=(nelx, nely))
left = PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0), tol=1e-9)
mid_right = NearPoint((float(nelx), nely / 2.0))
model = LinearElasticity(
    mesh,
    Material(E=1.0, nu=0.3, rho=1.0),
    supports=[(left, "all")],
    loads=[PointLoad(mid_right, force=(0.0, -1.0))],
)
chain = RadialDensityFilter(radius=2.4) | SIMP(p=3.0)
problem = Problem(
    model, chain, objective=Compliance(), constraints=[Volume() <= 0.4], optimizer=OC(move=0.2)
)
result = Study(problem, schedule=Schedule.default(max_iter=60, tol=1e-3)).run()
print(f"compliance {result.objective:.1f}, {result.iterations} iterations")

fig = result.view()
fig.savefig("cantilever.png", dpi=150, bbox_inches="tight")
```
