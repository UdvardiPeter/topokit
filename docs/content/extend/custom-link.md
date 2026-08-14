# Custom chain link

A link is a spec plus a bound form. `LinkSpec` subclasses hold the parameters and compose into a chain with `|`, the operator used throughout the tutorials (`DensityFilter(radius=1.5) | SIMP()`). `LinkSpec.build(mesh)` returns the bound form, `BoundLink`, which holds the mesh-dependent arrays. `BoundLink.apply` is the forward map; `BoundLink.pullback` is the vector-Jacobian product, taking a gradient with respect to the link's output and returning the gradient with respect to its input.

`Dilate` below is an elementwise density-biasing link, `rho -> rho**(1/k)`. It is not terminal, so it sits between the filter and the material interpolation in the chain.

```python
from dataclasses import dataclass

import numpy as np

from topokit import BoundLink, DensityFilter, LinkSpec, SIMP, StructuredGrid
from topokit.testing import assert_gradient_matches


@dataclass(frozen=True)
class Dilate(LinkSpec):
    """Elementwise rho -> rho**(1/k), k >= 1; biases the design toward material."""

    k: float = 2.0

    def __post_init__(self):
        if self.k < 1.0:
            raise ValueError("k must be >= 1")

    def build(self, mesh):
        return _BoundDilate(self.k)


class _BoundDilate(BoundLink):
    def __init__(self, k):
        self.k = k

    def apply(self, x):
        return x ** (1.0 / self.k)

    def pullback(self, x, grad_out):
        return grad_out * (1.0 / self.k) * np.maximum(x, 1e-12) ** (1.0 / self.k - 1.0)
```

A terminal link ends the chain (`SIMP` here, `RampConductivity` in [Custom physics](custom-physics.md)); its spec sets `is_terminal = True` and its bound form sets `out_field` to the `FieldSpec` the physics consumes. A reduced-input link goes first and shrinks the design space, the way symmetry does; its spec sets `is_reduced_input = True` and its bound form sets `n_reduced` to the size of that smaller space. `fd_example` is a classmethod needed only by links registered in the plugin registry, for the registry-wide gradient meta-test; a link used directly, like `Dilate` here, does not need it.

Bind the chain to a mesh and check the pullback against central differences, the same check every link in TopoKit passes.

```python
mesh = StructuredGrid(shape=(4, 2), spacing=(1.0, 1.0))
chain = DensityFilter(radius=1.5) | Dilate(k=2.0) | SIMP()
bound = chain.bind(mesh)

x = np.random.default_rng(5).uniform(0.2, 0.8, bound.n_vars)
w = np.random.default_rng(6).standard_normal(mesh.n_elements)
assert_gradient_matches(lambda xx: float(bound.apply(xx) @ w), lambda xx: bound.pullback(xx, w), x)
print("custom link composes and its gradient verifies")
```

## Next

[Custom physics](custom-physics.md) uses a terminal link like this one to interpolate a different physical property: thermal conductivity instead of stiffness.
