# Install

TopoKit is pre-alpha, published to PyPI as a dev release:

    pip install --pre topokit

Python 3.12 or newer. The core depends only on numpy and scipy.

| Extra | Adds | For |
| ----- | ---- | --- |
| `fast` | pyamg | AMG-preconditioned CG, the default for large 3D problems |
| `viz` | matplotlib, pyvista | `result.view()`, convergence plots, 3D iso-surfaces |
| `jax` | jax | JAX-backed assembly kernels via `use_backend` |
| `all` | all of the above | |

    pip install --pre 'topokit[all]'

Verify the install:

    python -c "import topokit; print(topokit.__version__)"
