# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""The base of every error TopoKit raises.

Each module keeps its own error type and its own builtin base (``MeshError``
is a ``ValueError``, ``SolverError`` a ``RuntimeError``), so code that
catches ``ValueError`` keeps working. ``TopoKitError`` sits alongside those
so a caller can wrap a whole run in one ``except`` without naming every
module, and so adding an error type later does not break that catch.

``FemError`` is the exception to "each module keeps its own": ``topokit.fem``
is a package whose element kernels raise it, and a submodule importing it
from the package root would close an import cycle. ``topokit.fem`` re-exports
it, so ``from topokit.fem import FemError`` still works.

This module has no imports and sits at the bottom of the layer DAG.
"""

from __future__ import annotations


class TopoKitError(Exception):
    """Base of every error raised by TopoKit.

    Examples
    --------
    Catch anything the library raises::

        try:
            result = Study(problem).run()
        except TopoKitError as exc:
            print(f"the run could not start or finish: {exc}")

    The specific types stay available (and keep their builtin bases), so
    ``except ProblemError`` and ``except ValueError`` are both unaffected.
    """


class FemError(TopoKitError, ValueError):
    """Invalid physics setup or input."""
