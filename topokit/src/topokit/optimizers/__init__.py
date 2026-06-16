# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Design-variable optimizers.

An optimizer is a pure stepper over bounded design variables. ``OC`` is the
optimality-criteria update for a single volume constraint; ``MMA`` is the
clean-room Method of Moving Asymptotes for general constraints.
"""

from topokit.optimizers._base import Optimizer, OptimizerError, StepResult
from topokit.optimizers._mma import MMA, MMASubproblem, SubproblemSolution, solve_subproblem
from topokit.optimizers._oc import OC

__all__ = [
    "MMA",
    "OC",
    "MMASubproblem",
    "Optimizer",
    "OptimizerError",
    "StepResult",
    "SubproblemSolution",
    "solve_subproblem",
]
