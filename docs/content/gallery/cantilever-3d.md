# 3D cantilever

A 3D cantilever, 24x12x12 elements: the left face is fully fixed, a downward point load at the center of the right face. Volume fraction 0.3.

![3D cantilever](img/cantilever-3d.png)

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

nelx, nely, nelz = 24, 12, 12
mesh = StructuredGrid.box(size=(float(nelx), float(nely), float(nelz)), shape=(nelx, nely, nelz))
left = PlaneSlab(point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), tol=1e-9)
tip = NearPoint((float(nelx), nely / 2.0, nelz / 2.0))
model = LinearElasticity(
    mesh,
    Material(E=1.0, nu=0.3, rho=1.0),
    supports=[(left, "all")],
    loads=[PointLoad(tip, force=(0.0, -1.0, 0.0))],
)
chain = RadialDensityFilter(radius=1.5) | SIMP(p=3.0)
problem = Problem(
    model, chain, objective=Compliance(), constraints=[Volume() <= 0.3], optimizer=OC(move=0.2)
)
result = Study(problem, schedule=Schedule.default(max_iter=60, tol=1e-3)).run()
print(f"compliance {result.objective:.1f}, {result.iterations} iterations")

plotter = result.view(off_screen=True)
plotter.screenshot("cantilever-3d.png")
```
