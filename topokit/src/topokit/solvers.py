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

import warnings
from typing import Any, Protocol, runtime_checkable

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from topokit.backend import SparseMatrix

AUTO_DIRECT_LIMIT = 150_000


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
    """Sparse direct solver: CHOLMOD when scikit-sparse is installed, else splu."""

    def __init__(self) -> None:
        self._solve_fn: Any = None

    def prepare(self, matrix: SparseMatrix) -> None:
        """Factorize ``matrix``."""
        csr = _to_scipy_csr(matrix)
        try:
            from sksparse.cholmod import cholesky  # type: ignore[import-not-found]

            self._solve_fn = cholesky(csr.tocsc())
        except ImportError:
            self._solve_fn = scipy.sparse.linalg.splu(csr.tocsc()).solve

    def solve(self, rhs: Any) -> Any:
        """Solve for ``rhs`` using the stored factorization."""
        if self._solve_fn is None:
            raise SolverError("call prepare() before solve()")
        return self._solve_fn(np.asarray(rhs, dtype=np.float64))


class AmgCG:
    """Conjugate gradients with a pyamg smoothed-aggregation preconditioner.

    ``tol`` is the relative residual tolerance. Requires the ``fast`` extra.
    """

    def __init__(self, tol: float = 1e-8, max_iter: int | None = None) -> None:
        self.tol = float(tol)
        self.max_iter = max_iter
        self._matrix: scipy.sparse.csr_array | None = None
        self._preconditioner: Any = None

    def prepare(self, matrix: SparseMatrix) -> None:
        """Build the AMG hierarchy for ``matrix``."""
        try:
            pyamg = _load_pyamg()
        except ImportError as exc:
            raise SolverError(
                "AmgCG needs pyamg; install the extra: pip install topokit[fast]"
            ) from exc
        csr = _to_scipy_csr(matrix)
        self._matrix = csr
        self._preconditioner = pyamg.smoothed_aggregation_solver(
            scipy.sparse.csr_matrix(csr)
        ).aspreconditioner()

    def solve(self, rhs: Any) -> Any:
        """Run preconditioned CG per right-hand side."""
        if self._matrix is None:
            raise SolverError("call prepare() before solve()")
        b = np.asarray(rhs, dtype=np.float64)
        single = b.ndim == 1
        cols = b[:, None] if single else b
        out = np.empty_like(cols)
        for j in range(cols.shape[1]):
            x, info = scipy.sparse.linalg.cg(
                self._matrix,
                cols[:, j],
                rtol=self.tol,
                atol=0.0,
                maxiter=self.max_iter,
                M=self._preconditioner,
            )
            if info != 0:
                raise SolverError(
                    f"CG did not converge (info={info}, tol={self.tol}, maxiter={self.max_iter})"
                )
            out[:, j] = x
        return out[:, 0] if single else out


def auto_solver(n_dof: int) -> LinearSolver:
    """Doc-04 selection rule: Direct below 150k DOF, AmgCG above when available."""
    if n_dof < AUTO_DIRECT_LIMIT:
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
