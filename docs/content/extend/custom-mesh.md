# Custom mesh

Everything in TopoKit that reads geometry reads the `Mesh` protocol: selectors, filters, responses, the physics models, the writers. Satisfy it and the library does not care where your elements came from. The protocol has 15 members, but 7 of them follow mechanically from the other 8, so `MeshBase` (in `topokit.mesh`) implements the 7 and leaves you the 8. `StructuredGrid` and `UnstructuredMesh` are both built on it.

| You provide                                                  | `MeshBase` derives                                          |
| ------------------------------------------------------------ | ----------------------------------------------------------- |
| `nodes`, `element_nodes`                                     | `n_nodes`, `n_elements`, `dim`                              |
| `element_kind`, `element_volumes`, `element_centroids`        | `design` (neither solid nor void), `active_elements` (not void) |
| the masks, by calling `_set_masks(solid, void)` in `__init__` | `solid`, `void`, validated and write-protected               |
| `_compute_boundary_faces()`, called once                     | `active_nodes`, `node_index_map`, cached `boundary_faces()`  |

Before writing one: if your mesh is a plain array pair, `UnstructuredMesh(nodes, elements, "tri3")` already is that class, and `UnstructuredMesh.from_file` reads 20-odd formats through meshio. Subclass `MeshBase` when the arrays are generated, when they live in another structure you would rather not copy, or when you know something the general code has to discover — the example below knows its own boundary, so it skips the facet pairing search entirely.

## An annulus of triangles

`RingMesh` generates a tri3 annulus from four numbers. The connectivity wraps around in the angular direction, so the boundary is exactly the two rims; `_compute_boundary_faces` writes them out directly. Face nodes are in perimeter order and the outward normal is explicit, which is all `SurfaceTraction` and the face selectors need.

```python
import numpy as np

from topokit import (
    OC,
    SIMP,
    Compliance,
    DensityFilter,
    Material,
    PointLoad,
    Problem,
    Schedule,
    Study,
    UnstructuredElasticity,
    Volume,
)
from topokit.mesh import BoundaryFaces, Mesh, MeshBase
from topokit.selection import Box, NearPoint


class RingMesh(MeshBase):
    """An annulus of tri3 elements, generated from four numbers."""

    def __init__(self, r_inner: float, r_outer: float, n_radial: int, n_angular: int) -> None:
        self.n_radial, self.n_angular = n_radial, n_angular
        radii = np.linspace(r_inner, r_outer, n_radial + 1)
        angles = 2.0 * np.pi * np.arange(n_angular) / n_angular
        self._nodes = np.stack(
            [
                (radii[:, None] * np.cos(angles)[None, :]).ravel(),
                (radii[:, None] * np.sin(angles)[None, :]).ravel(),
            ],
            axis=1,
        )
        ring, sector = np.divmod(np.arange(n_radial * n_angular), n_angular)
        inner = ring * n_angular + sector
        outer = inner + n_angular
        next_sector = (sector + 1) % n_angular
        inner_next = ring * n_angular + next_sector
        outer_next = inner_next + n_angular
        # two triangles per cell, counterclockwise: ids 2*cell and 2*cell + 1
        self._elements = np.stack(
            [
                np.stack([inner, outer, outer_next], axis=1),
                np.stack([inner, outer_next, inner_next], axis=1),
            ],
            axis=1,
        ).reshape(-1, 3)
        self._set_masks(None, None)

    @property
    def element_kind(self) -> str:
        return "tri3"

    @property
    def nodes(self) -> np.ndarray:
        return self._nodes

    @property
    def element_nodes(self) -> np.ndarray:
        return self._elements

    @property
    def element_volumes(self) -> np.ndarray:
        p = self._nodes[self._elements]
        e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
        return np.asarray(0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]))

    @property
    def element_centroids(self) -> np.ndarray:
        return np.asarray(self._nodes[self._elements].mean(axis=1))

    def _compute_boundary_faces(self) -> BoundaryFaces:
        # the generator already knows the boundary: the two rims. No facet
        # pairing search, which is the reason this mesh is worth writing.
        sector = np.arange(self.n_angular)
        next_sector = (sector + 1) % self.n_angular
        last = self.n_radial * self.n_angular
        edges = np.concatenate(
            [
                np.stack([sector, next_sector], axis=1),  # inner rim
                np.stack([last + sector, last + next_sector], axis=1),  # outer rim
            ]
        )
        owner = np.concatenate(
            [
                2 * sector + 1,  # the inner rim edge belongs to each cell's 2nd triangle
                2 * ((self.n_radial - 1) * self.n_angular + sector),  # ... and the 1st outside
            ]
        )
        coords = self._nodes[edges]
        centroid = coords.mean(axis=1)
        radial = centroid / np.linalg.norm(centroid, axis=1)[:, None]
        sign = np.concatenate([-np.ones(self.n_angular), np.ones(self.n_angular)])
        return BoundaryFaces(
            owner=owner,
            normal=sign[:, None] * radial,
            area=np.linalg.norm(coords[:, 1] - coords[:, 0], axis=1),
            centroid=centroid,
            nodes=edges,
        )


mesh = RingMesh(r_inner=4.0, r_outer=10.0, n_radial=6, n_angular=48)
assert isinstance(mesh, Mesh)
print(f"{mesh.n_elements} elements, {mesh.n_nodes} nodes, dim {mesh.dim}")
print(f"area {mesh.element_volumes.sum():.3f} vs exact {np.pi * (10.0**2 - 4.0**2):.3f}")
print(f"boundary length {mesh.boundary_faces().area.sum():.3f} vs exact {2 * np.pi * 14.0:.3f}")

# the derived half, which MeshBase filled in
print(f"design elements {int(mesh.design.sum())}, active nodes {int(mesh.active_nodes.sum())}")
```

The measures land within the discretization error of the exact annulus, which is the first thing to check on a new mesh: an area that is off by a factor, or a negative element volume, means the connectivity winding is wrong, and every force and compliance downstream will be wrong with it.

## Running a study on it

Nothing else changes. `UnstructuredElasticity` is written against the protocol, not against `UnstructuredMesh`, so any mesh whose `element_kind` is `tri3`, `tet4` or `tet10` works; a kind it has no element for is rejected at construction. Selectors resolve against the node coordinates and face centroids the mesh reports, so the bottom of the ring is clamped and the top is pushed sideways.

```python
model = UnstructuredElasticity(
    mesh,
    Material(E=1.0, nu=0.3, rho=1.0),
    supports=[(Box((-10.0, -10.0), (10.0, -4.0), tol=0.5), "all")],
    loads=[PointLoad(NearPoint((0.0, 10.0)), (1.0, 0.0))],
)
problem = Problem(
    model,
    DensityFilter(radius=1.0) | SIMP(p=3.0),
    objective=Compliance(),
    constraints=[Volume() <= 0.5],
    optimizer=OC(),
)
result = Study(problem, schedule=Schedule.single(max_iter=15, tol=1e-3)).run()
print(f"compliance {result.history['objective'][0]:.4g} -> {result.objective:.4g}")
```

`DensityFilter` falls back to its sparse-matrix path on a non-grid mesh, built from element centroids, so it needs nothing from the mesh either.

## What the base will not do for you

- **Ordering conventions.** `element_nodes` must follow the VTK node order for its `element_kind`, and simplex elements must have positive signed area or volume. Nothing here checks that; `UnstructuredMesh`'s constructor does, so a mesh of simplices that can afford one array copy can validate itself by building one.
- **Immutability.** `StructuredGrid` and `UnstructuredMesh` write-protect every derived array (`arr.flags.writeable = False`) so a caller cannot corrupt a mesh another object is holding. The base does that for the masks it validates and for nothing else.
- **Face-id stability.** `boundary_faces()` is cached, so ids are stable for the lifetime of the instance. If ids must also be stable across two instances built from the same input — a checkpoint resumed later, say — your face order has to be deterministic; `UnstructuredMesh` gets that by ordering faces by their sorted corner-node tuple.

## Next

[Custom physics](custom-physics.md) does the same thing one layer up: a `PhysicsModel` from scratch, and `ElasticityBase` for the case where you only want to change the element.
