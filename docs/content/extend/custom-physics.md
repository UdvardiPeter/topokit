# Custom physics

A full worked physics module outside elasticity: heat conduction, with a RAMP terminal link and stock `Compliance` reused unmodified as thermal compliance. Nothing in `topokit`'s core changes; both pieces are ordinary implementations of the `LinkSpec` and `PhysicsModel` extension points, the same ones [Custom chain link](custom-link.md) and [Extend](index.md) describe.

## A terminal link for conductivity

`RampConductivity` is a RAMP-style interpolation (Rational Approximation of Material Properties), a common SIMP alternative that keeps a nonzero gradient at `rho = 0`. It plays the terminal role `SIMP` plays for elasticity: `is_terminal = True` on the spec, `out_field` on the bound form, here a new `FieldSpec("conductivity_scale")` rather than `SIMP`'s `stiffness_scale`.

```python
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from topokit import (
    MMA,
    BoundLink,
    Compliance,
    DensityFilter,
    FieldSpec,
    LinkSpec,
    Problem,
    Schedule,
    Solution,
    StructuredGrid,
    Study,
    Volume,
)
from topokit.backend import SparseMatrix, active_backend
from topokit.testing import assert_gradient_matches

CONDUCTIVITY = FieldSpec("conductivity_scale")


@dataclass(frozen=True)
class RampConductivity(LinkSpec):
    """RAMP-style terminal interpolation for conduction."""

    q: float = 3.0
    scale_min: float = 1e-3
    is_terminal: ClassVar[bool] = True

    def build(self, mesh):
        return _BoundRamp(self.q, self.scale_min)


class _BoundRamp(BoundLink):
    out_field = CONDUCTIVITY

    def __init__(self, q, smin):
        self.q, self.smin = q, smin

    def apply(self, x):
        return self.smin + (1.0 - self.smin) * x / (1.0 + self.q * (1.0 - x))

    def pullback(self, x, g):
        d = (1.0 + self.q) / (1.0 + self.q * (1.0 - x)) ** 2
        return g * (1.0 - self.smin) * d
```

## The physics model

A `PhysicsModel` is any object with the protocol's methods (`assemble`, `loads`, `element_energies`, `element_stress`) and an `expected_field` matching the chain's terminal `out_field`; a mismatch fails at `Problem` construction, not partway through a solve. `HeatConduction` below assembles a per-element 4x4 conduction stiffness matrix, fixes the temperature on the mesh's left edge, and puts a unit heat sink at mid-right. The boundary-condition reduction is hand-rolled here in about fifteen lines: build a free-DOF index map over the mesh's nodes, then drop the fixed rows and columns from the element connectivity before assembly.

```python
class HeatConduction:
    """2D conduction on a structured grid: temperature fixed on the left edge,
    a unit heat sink at mid-right, one degree of freedom per node."""

    expected_field: ClassVar[FieldSpec] = CONDUCTIVITY

    def __init__(self, mesh: StructuredGrid) -> None:
        self._mesh = mesh
        hx, hy = mesh.spacing
        r = hx / hy
        a, b = r / 3.0, 1.0 / (3.0 * r)
        self._ke = np.array(
            [
                [a + b, -a + b / 2, -a / 2 - b / 2, a / 2 - b],
                [-a + b / 2, a + b, a / 2 - b, -a / 2 - b / 2],
                [-a / 2 - b / 2, a / 2 - b, a + b, -a + b / 2],
                [a / 2 - b, -a / 2 - b / 2, -a + b / 2, a + b],
            ]
        )
        nx = mesh.shape[0]
        fixed = np.zeros(mesh.n_nodes, dtype=bool)
        fixed[:: nx + 1] = True
        self._free = np.full(mesh.n_nodes, -1, dtype=np.int64)
        self._free[~fixed] = np.arange((~fixed).sum())
        self._n_dof = int((~fixed).sum())
        edofs = self._free[mesh.element_nodes]
        rows = np.repeat(edofs, 4, axis=1).ravel()
        cols = np.tile(edofs, (1, 4)).ravel()
        keep = (rows >= 0) & (cols >= 0)
        self._rows, self._cols, self._keep = rows[keep], cols[keep], keep
        f = np.zeros(self._n_dof)
        sink = int(nx + (nx + 1) * (mesh.shape[1] // 2))
        f[self._free[sink]] = 1.0
        self._loads = f[:, None]

    @property
    def mesh(self):
        return self._mesh

    @property
    def n_dof(self):
        return self._n_dof

    @property
    def n_cases(self):
        return 1

    def assemble(self, scale: Any) -> SparseMatrix:
        vals = np.einsum("e,k->ek", np.asarray(scale), self._ke.ravel()).ravel()[self._keep]
        return active_backend().coo_to_csr(
            self._rows, self._cols, vals, shape=(self._n_dof, self._n_dof)
        )

    def loads(self):
        return self._loads

    def element_energies(self, u, scale):
        uu = np.append(np.asarray(u), 0.0)
        idx = np.where(self._free[self._mesh.element_nodes] >= 0,
                       self._free[self._mesh.element_nodes], self._n_dof)
        ue = uu[idx]
        return np.einsum("ei,ij,ej->e", ue, self._ke, ue) * np.asarray(scale)

    def element_stress(self, u, scale):
        return np.zeros(self._mesh.n_elements)
```

`Compliance` works untouched because it only calls `element_energies`. The self-adjoint contract it relies on is that `element_energies(u, ones)` returns the per-element quadratic form `u_e^T K_e u_e`; `HeatConduction.element_energies` supplies exactly that (`scale` folds the interpolated conductivity into the assembly path, `ones` gives the unscaled sensitivity kernel `Compliance` needs).

## Running it

The chain terminates in `RampConductivity`, its `out_field` matches `HeatConduction.expected_field`, and the rest is the elasticity pattern from the tutorials: a volume constraint, `MMA`, a `Schedule`, a `Study`.

```python
mesh = StructuredGrid.box(size=(20.0, 20.0), shape=(20, 20))
problem = Problem(
    HeatConduction(mesh),
    DensityFilter(radius=1.5) | RampConductivity(),
    objective=Compliance(),
    constraints=[Volume() <= 0.3],
    optimizer=MMA(),
)
result = Study(problem, schedule=Schedule.single(max_iter=50, tol=1e-3)).run()
print(f"thermal compliance {result.history['objective'][0]:.2f} -> {result.objective:.2f}")
```

## Verifying the gradient through model and solver

This gradient runs through the link, the physics model, and a real linear solve, unlike a density-only response that can be checked without one. Bind the chain, solve `HeatConduction`'s system at a design point on a small mesh, and check `Compliance`'s value and gradient (routed back through the chain's pullback) against central differences.

```python
from topokit.solvers import Direct

small = StructuredGrid.box(size=(5.0, 4.0), shape=(5, 4))
m2 = HeatConduction(small)
b2 = (DensityFilter(radius=1.5) | RampConductivity()).bind(small)
solver = Direct()


def solve(xx):
    ev = b2.evaluate(xx)
    solver.prepare(m2.assemble(ev.field))
    u = np.atleast_2d(np.asarray(solver.solve(m2.loads()))).reshape(m2.n_dof, -1)
    return ev, Solution(model=m2, mesh=small, displacements=u,
                        interpolated=ev.field, density=ev.density)


assert_gradient_matches(
    lambda xx: Compliance().value(solve(xx)[1]),
    lambda xx: solve(xx)[0].pullback(Compliance().grad_field(solve(xx)[1])),
    np.random.default_rng(9).uniform(0.3, 0.7, b2.n_vars),
)
print("thermal compliance gradient verified through chain, model, and solver")
```

## Next

[Backends and kernels](backends.md) covers the array-compute layer `assemble` and `coo_to_csr` run through, and how a third-party backend distributes itself.
