# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Linear solvers for the assembled free-DOF systems.

Both solvers assume symmetric positive definite systems, which the
assembled stiffness matrices are.

``Direct`` factorizes once and solves any number of right-hand sides;
``AmgCG`` runs conjugate gradients with a pyamg smoothed-aggregation
preconditioner (the path to large 3D systems). ``auto_solver`` picks per
the doc-04 rule: direct below 150k DOF, AMG-preconditioned CG above.

Solver accuracy bounds sensitivity accuracy: in the SIMP regime the system
condition number is roughly the stiffness contrast (1e9 with the default
floor), and the relative error of any solve is amplified accordingly. The
default CG tolerance of 1e-8 keeps the measured displacement error around
1e-5 there; tighten it before tightening optimizer convergence.
"""

from __future__ import annotations

import contextlib
import warnings
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import scipy.sparse
import scipy.sparse.linalg

from topokit.backend import SparseMatrix

_F64 = npt.NDArray[np.float64]

AUTO_DIRECT_LIMIT_2D = 150_000
AUTO_DIRECT_LIMIT_3D = 15_000


class SolverError(RuntimeError):
    """Solver misuse or numerical failure."""


@runtime_checkable
class LinearSolver(Protocol):
    """Factorize-then-solve contract; ``prepare`` is reusable across RHS."""

    def prepare(self, matrix: SparseMatrix) -> None:
        """Factorize or build the preconditioner for ``matrix``."""
        ...

    def solve(self, rhs: Any) -> Any:
        """Solve for one ``(n,)`` or several ``(n, k)`` right-hand sides."""
        ...


def _to_scipy_csr(matrix: SparseMatrix) -> scipy.sparse.csr_array:
    indptr, indices, data = matrix.csr_arrays()
    return scipy.sparse.csr_array(
        (np.asarray(data), np.asarray(indices), np.asarray(indptr)), shape=matrix.shape
    )


def _load_pyamg() -> Any:
    import pyamg  # type: ignore[import-untyped]

    return pyamg


class Direct:
    """Sparse direct solver: CHOLMOD when scikit-sparse is installed, else splu.

    SuperLU factorizes numerically singular systems without complaint and
    returns garbage, so every solve is residual-checked by default:
    under-constrained models fail loudly instead of producing huge
    plausible-looking displacements. ``residual_check=None`` disables it.
    """

    def __init__(self, residual_check: float | None = 1e-3) -> None:
        if residual_check is not None and residual_check <= 0.0:
            raise SolverError(f"residual_check must be > 0 or None, got {residual_check}")
        self.residual_check = residual_check
        self._solve_fn: Any = None
        self._matrix: scipy.sparse.csr_array | None = None

    def prepare(self, matrix: SparseMatrix) -> None:
        """Factorize ``matrix``."""
        csr = _to_scipy_csr(matrix)
        self._matrix = csr
        factorize: Any = None
        with contextlib.suppress(ImportError):
            from sksparse.cholmod import cholesky  # type: ignore[import-not-found]

            factorize = cholesky
        try:
            if factorize is not None:
                self._solve_fn = factorize(csr.tocsc())
            else:
                self._solve_fn = scipy.sparse.linalg.splu(csr.tocsc()).solve
        except Exception as exc:  # CHOLMOD raises its own hierarchy, splu RuntimeError
            raise SolverError(f"factorization failed: {exc}; check supports") from exc

    def solve(self, rhs: Any) -> Any:
        """Solve for ``rhs`` using the stored factorization."""
        if self._solve_fn is None or self._matrix is None:
            raise SolverError("call prepare() before solve()")
        b = np.asarray(rhs, dtype=np.float64)
        x = self._solve_fn(b)
        if self.residual_check is not None:
            cols_b = b[:, None] if b.ndim == 1 else b
            cols_x = x[:, None] if x.ndim == 1 else x
            num = np.linalg.norm(cols_b - self._matrix @ cols_x, axis=0)
            den = np.maximum(np.linalg.norm(cols_b, axis=0), 1e-300)
            worst = float((num / den).max())
            if worst > self.residual_check:
                raise SolverError(
                    f"relative residual {worst:.2e} exceeds {self.residual_check:g}; "
                    "the system may be singular - check supports"
                )
        return x


class _IterCounter:
    """Counts CG iterations via the scipy callback protocol."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self, _xk: Any) -> None:
        self.count += 1


class AmgCG:
    """Conjugate gradients with a pyamg smoothed-aggregation preconditioner.

    ``tol`` is the relative residual tolerance. Requires the ``fast`` extra.
    """

    def __init__(self, tol: float = 1e-8, max_iter: int | None = None) -> None:
        if tol <= 0.0:
            raise SolverError(f"tol must be > 0, got {tol}")
        if max_iter is not None and max_iter < 1:
            raise SolverError(f"max_iter must be >= 1, got {max_iter}")
        self.tol = float(tol)
        self.max_iter = max_iter
        self.last_iterations: int = 0
        self._matrix: scipy.sparse.csr_array | None = None
        self._preconditioner: Any = None
        self._near_nullspace: _F64 | None = None

    def set_near_nullspace(self, modes: Any) -> None:
        """Provide near-nullspace modes (e.g. rigid-body modes) for the hierarchy.

        Cuts elasticity CG iteration counts typically 2-3x versus pyamg's
        default constant-vector hint. The orchestration layer wires this
        automatically when the physics model can supply modes.
        """
        b = np.asarray(modes, dtype=np.float64)
        if b.ndim != 2 or b.shape[1] < 1:
            raise SolverError(f"modes must have shape (n_dof, k), got {b.shape}")
        self._near_nullspace = b

    def prepare(self, matrix: SparseMatrix) -> None:
        """Build the AMG hierarchy for ``matrix``."""
        try:
            pyamg = _load_pyamg()
        except ImportError as exc:
            raise SolverError(
                "AmgCG needs pyamg; install the extra: pip install topokit[fast]"
            ) from exc
        csr = _to_scipy_csr(matrix)
        b = self._near_nullspace
        if b is not None and b.shape[0] != csr.shape[0]:
            raise SolverError(
                f"near-nullspace has {b.shape[0]} rows for a {csr.shape[0]}-DOF system"
            )
        self._matrix = csr
        # pyamg estimates spectral radii with numpy's global RNG; seed it so
        # hierarchies (and therefore runs) are reproducible, then restore.
        state = np.random.get_state()
        np.random.seed(0)
        try:
            self._preconditioner = pyamg.smoothed_aggregation_solver(
                scipy.sparse.csr_matrix(csr), B=b
            ).aspreconditioner()
        finally:
            np.random.set_state(state)

    def solve(self, rhs: Any) -> Any:
        """Run preconditioned CG per right-hand side."""
        if self._matrix is None:
            raise SolverError("call prepare() before solve()")
        self.last_iterations = 0  # reset up front; a mid-solve failure leaves no stale count
        b = np.asarray(rhs, dtype=np.float64)
        single = b.ndim == 1
        cols = b[:, None] if single else b
        out = np.empty_like(cols)
        worst = 0
        for j in range(cols.shape[1]):
            counter = _IterCounter()
            x, info = scipy.sparse.linalg.cg(
                self._matrix,
                cols[:, j],
                rtol=self.tol,
                atol=0.0,
                maxiter=self.max_iter,
                M=self._preconditioner,
                callback=counter,
            )
            if info != 0:
                raise SolverError(
                    f"CG did not converge (info={info}, tol={self.tol}, maxiter={self.max_iter})"
                )
            out[:, j] = x
            worst = max(worst, counter.count)
        self.last_iterations = worst
        return out[:, 0] if single else out


def auto_solver(n_dof: int, dim: int) -> LinearSolver:
    """Pick a solver by size and dimension.

    Direct factorization fill-in makes the crossover dimension-dependent:
    measured on 3D elasticity, AMG-preconditioned CG already wins at 14k
    DOF, while 2D direct stays competitive far longer.
    """
    if dim not in (2, 3):
        raise SolverError(f"dim must be 2 or 3, got {dim}")
    limit = AUTO_DIRECT_LIMIT_2D if dim == 2 else AUTO_DIRECT_LIMIT_3D
    if n_dof < limit:
        return Direct()
    try:
        _load_pyamg()
    except ImportError:
        warnings.warn(
            f"{n_dof} DOF exceeds the direct-solver limit and pyamg is not "
            "installed; falling back to Direct. Install it: pip install topokit[fast]",
            stacklevel=2,
        )
        return Direct()
    return AmgCG()
