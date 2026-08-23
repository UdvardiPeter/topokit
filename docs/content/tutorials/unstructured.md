# Unstructured meshes

The tutorials so far mesh a box with `StructuredGrid`. `UnstructuredMesh`
covers everything else: an imported CAD geometry, an arbitrary boundary, a
mesh gmsh generated. This tutorial builds one from plain numpy arrays and
runs the same six-object pipeline (mesh, physics, chain, problem, study,
result) as [your first 2D part](first-2d.md); only the mesh and physics
model swap in.

## Mesh

`UnstructuredMesh` takes node coordinates and element connectivity as plain
numpy arrays, the `Mesh` protocol's minimal case: any tool that produces
`(nodes, elements)` arrays works, from a hand-rolled triangulation to a full
CAD mesher. The triangulated grid below is just an easy in-page example,
splitting each quad of a regular grid into two triangles so the tutorial
needs no extra dependency to run.

```python
import numpy as np

from topokit import (
    OC,
    SIMP,
    Compliance,
    DensityFilter,
    Material,
    NearPoint,
    PlaneSlab,
    PointLoad,
    Problem,
    Schedule,
    Study,
    UnstructuredElasticity,
    UnstructuredMesh,
    Volume,
)

nx, ny = 24, 8
xs, ys = np.meshgrid(np.linspace(0.0, 12.0, nx + 1), np.linspace(0.0, 4.0, ny + 1), indexing="ij")
nodes = np.column_stack([xs.ravel(), ys.ravel()])
quads = np.array(
    [
        [i * (ny + 1) + j, (i + 1) * (ny + 1) + j, (i + 1) * (ny + 1) + j + 1, i * (ny + 1) + j + 1]
        for i in range(nx)
        for j in range(ny)
    ]
)
elements = np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]]).astype(np.int64)
mesh = UnstructuredMesh(nodes, elements, "tri3")
print(f"{mesh.n_elements} triangles, {mesh.n_nodes} nodes")
```

## Physics

`UnstructuredElasticity` mirrors `LinearElasticity`'s constructor exactly:
the same `Material`, the same geometric `supports`/`loads` selection through
`PlaneSlab` and `NearPoint`. What changes underneath is the per-element
stiffness: simplex elements aren't geometrically identical across the mesh
the way a structured grid's cells are, so each element gets its own
stiffness matrix instead of one shared template.

```python
model = UnstructuredElasticity(
    mesh,
    Material(E=1.0, nu=0.3, rho=1.0),
    supports=[(PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0)), "all")],
    loads=[PointLoad(NearPoint((12.0, 2.0)), force=(0.0, -1.0))],
)
```

## Problem and study

The chain, `Problem`, and `Study` are mesh-agnostic; nothing here changes
from the structured tutorials.

```python
problem = Problem(
    model,
    DensityFilter(radius=1.2) | SIMP(p=3.0),
    objective=Compliance(),
    constraints=[Volume() <= 0.5],
    optimizer=OC(),
)
result = Study(problem, schedule=Schedule.single(p=3.0, max_iter=40, tol=1e-3)).run()
print(f"compliance {result.history['objective'][0]:.1f} -> {result.objective:.1f}")
```

## Viewing the result

`result.view()` renders the triangulation directly: a matplotlib
`tripcolor` plot in 2D, colored by each triangle's density. In 3D, an
unstructured mesh's view returns a PyVista scene showing the cells at or
above the `iso` threshold rather than a contoured surface, since a simplex
mesh has no implicit smooth field to contour the way a structured grid's
iso-surface does.

```python
fig = result.view()
fig.savefig("part.png", dpi=150)
```

## Importing real meshes

Hand-building arrays works for a tutorial; real geometry comes from a
mesher. `UnstructuredMesh.from_file` reads gmsh and other formats through
meshio, installed with the `mesh` extra: `pip install --pre
'topokit[mesh]'`. gmsh physical-group names resolve directly to `solid`/
`void` regions: pass the group name and it's matched against the mesh's
elements automatically. `groups` on the resulting mesh carries every named
region, not just the ones picked as `solid` or `void`.

<!-- no-run -->
```python
mesh = UnstructuredMesh.from_file("bracket.msh", solid="fixture", void="hole")
```

## Supported elements

`UnstructuredMesh` supports three element kinds, chosen by the
`element_kind` argument: `tri3` (2D triangles, used above), `tet4` (3D
tetrahedra), and `tet10` (quadratic tetrahedra, curved-edge accuracy at
higher cost). `UnstructuredElasticity` runs on all three.

## What's still structured-only

`DensityFilter` and `RadialDensityFilter` work unchanged on an unstructured
mesh: `radius` stays a physical length, and both switch to an exact
sparse-matrix implementation internally rather than the structured grid's
separable convolution. `SymmetryMap`, `view_slices`, and `LiveView` are
structured-grid-only for now; each raises rather than running silently
wrong on an unstructured mesh.

## Next

[Custom physics](../extend/custom-physics.md) extends the same
`Mesh`/`PhysicsModel` protocol seam this tutorial uses, with a worked
heat-conduction example outside elasticity. The [Gallery](../gallery/index.md)
has full-size results built the same way.
