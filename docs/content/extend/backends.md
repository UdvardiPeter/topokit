# Backends and kernels

`ArrayBackend` (`topokit.backend`) is the protocol the numerics modules are written against instead of importing numpy directly: `asarray`, `zeros`, `einsum`, `scatter_add`, `gather`, `coo_to_csr`, `csr_from_parts`. Numpy is the built-in default and the process-wide fallback outside any explicit selection.

`use_backend(...)` is a context manager that selects the active backend for the dynamic extent of a block. It accepts a backend instance or a name string, and names resolve through the plugin registry's `backends` group. The selection governs resolution at call time, not where an object was constructed: a mesh, chain, or model built outside `use_backend` still runs its kernels under whichever backend is active when `apply`, `pullback`, or `assemble` actually runs.

Some kernels have backend-specific implementations for performance; `get_kernel(name)` resolves one at call time, first for the active backend by name, falling back to the generic implementation shared by every backend. A backend only needs to register a specialized kernel where it has a faster path; everything else falls through to `get_kernel`'s generic fallback.

The built-in JAX backend (`pip install topokit[jax]`) is the reference implementation for a third-party backend. Its class, `JaxBackend`, and its module-level instance, `BACKEND`, live in `topokit/src/topokit/jax.py`; dense array ops and the assembly kernel run on JAX arrays there, while sparse matrix construction and the linear solve stay on the host.

A new backend distributes itself the same way: a module-level `ArrayBackend` instance registered under the `topokit.backends` entry-point group in its package's `pyproject.toml`, the same pattern the JAX backend uses (`jax = "topokit.jax:BACKEND"`). Once installed, `use_backend("your-backend-name")` resolves it without an explicit import.

Correctness is defined by the backend conformance suite, `ArrayBackendConformance` in `topokit/src/topokit/backend/conformance.py`: every backend, built-in or third-party, is expected to pass it. `topokit/tests/test_backend.py` subclasses it for the numpy backend, and `topokit/tests/test_jax.py` subclasses it again for the JAX backend; a new backend runs the same suite against its own instance the same way. The suite grows with the protocol, so a passing subclass after a `topokit` upgrade confirms the backend still satisfies the contract.
