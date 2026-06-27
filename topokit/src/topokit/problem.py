# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Problem assembly and the optimization loop (orchestration layer).

``Problem`` validates and holds the pieces of an optimization (physics
model, parametrization chain, objective, constraints, optimizer, solver).
``Study`` runs the loop: map design variables through the chain, solve the
physics, evaluate the objective and constraints, route their gradients back
to the design variables, step the optimizer, emit events, and check
convergence. This is the only module that knows the algorithm's plot;
everything it calls is a protocol.

The objective is normalized by its initial value before being handed to the
optimizer: MMA is not scale-invariant and needs it, OC is unaffected.
Gradients route to the design variables by each response's ``field_basis``
(the interpolated stiffness scale through ``chain.pullback``, the physical
density through ``chain.pullback_density``).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt

from topokit.events import (
    EventBus,
    FieldSnapshot,
    IterationFinished,
    StageFinished,
    StudyFinished,
    StudyStarted,
)
from topokit.fem import PhysicsModel
from topokit.fields import DesignField
from topokit.mesh import StructuredGrid
from topokit.optimizers import OC, Optimizer
from topokit.parametrization import SIMP, BoundChain, Chain, Heaviside
from topokit.responses import Constraint, Response, Solution
from topokit.solvers import LinearSolver, auto_solver

_F64 = npt.NDArray[np.float64]


class ProblemError(ValueError):
    """Invalid problem assembly or study configuration."""


def _bind_chain(chain: Chain | BoundChain, model: PhysicsModel) -> BoundChain:
    if isinstance(chain, BoundChain):
        return chain
    return chain.bind(model.mesh)


def _staged_chain(spec: Chain, mesh: StructuredGrid, *, p: float, beta: float) -> BoundChain:
    """Rebind ``spec`` with SIMP ``p`` and Heaviside ``beta`` replaced (Option A).

    Continuation overrides the authored ``SIMP.p``/``Heaviside.beta`` with the
    stage values; other link types pass through. ``replace`` re-runs each
    link's ``__post_init__``, so a bad staged value fails loud before the stage.
    """
    links = tuple(
        replace(link, p=p)
        if isinstance(link, SIMP)
        else replace(link, beta=beta)
        if isinstance(link, Heaviside)
        else link
        for link in spec.links
    )
    return Chain(links).bind(mesh)


class Problem:
    """A validated topology-optimization problem ready to run."""

    def __init__(
        self,
        model: PhysicsModel,
        chain: Chain | BoundChain,
        objective: Response,
        constraints: list[Constraint] | tuple[Constraint, ...] = (),
        optimizer: Optimizer | None = None,
        solver: LinearSolver | str = "auto",
    ) -> None:
        self.model = model
        self.chain = _bind_chain(chain, model)
        if self.chain.out_field != model.expected_field:
            raise ProblemError(
                f"chain output field {self.chain.out_field} does not match the physics "
                f"expected field {model.expected_field}"
            )
        self.objective = objective
        self.constraints = tuple(constraints)
        # Events and history key responses by name; collisions would silently
        # drop a constraint from the reporting, so reject them up front. The
        # objective lives under "objective"; "change" is the step-size series.
        reserved = {"objective", "change", objective.name}
        seen: set[str] = set()
        for c in self.constraints:
            key = c.report_key
            if key in reserved or key in seen:
                raise ProblemError(
                    f"constraint report key {key!r} collides with the objective, another "
                    f"constraint, or a reserved history key; give the constraint a distinct "
                    f"label via Constraint.labeled(...)"
                )
            seen.add(key)
        self.optimizer = optimizer if optimizer is not None else OC()
        if isinstance(solver, str):
            if solver != "auto":
                raise ProblemError(f"unknown solver {solver!r}; use 'auto' or a LinearSolver")
            self.solver: LinearSolver = auto_solver(model.n_dof, model.mesh.dim)
        else:
            self.solver = solver

    def default_volume_fraction(self) -> float:
        """Guess an initial design density from the first volume constraint."""
        for c in self.constraints:
            if c.response.name == "volume" and c.sense == "<=":
                return float(c.bound)
        return 0.5


@dataclass(frozen=True)
class Stage:
    """One continuation stage.

    Holds the SIMP ``p`` and Heaviside ``beta`` for the stage plus the
    per-stage convergence cap/tol (E6 ``200/stage``).
    """

    p: float
    beta: float
    max_iter: int = 200
    tol: float = 0.01

    def __post_init__(self) -> None:
        if self.p < 1.0:
            raise ProblemError(f"stage p must be >= 1, got {self.p}")
        if self.beta <= 0.0:
            raise ProblemError(f"stage beta must be > 0, got {self.beta}")
        if self.max_iter < 1:
            raise ProblemError(f"stage max_iter must be >= 1, got {self.max_iter}")
        if self.tol < 0.0:
            raise ProblemError(f"stage tol must be >= 0, got {self.tol}")


@dataclass(frozen=True)
class Schedule:
    """An ordered tuple of continuation stages (E7, introspectable/replaceable)."""

    stages: tuple[Stage, ...]

    def __post_init__(self) -> None:
        if not self.stages:
            raise ProblemError("schedule stages must not be empty")

    @classmethod
    def default(cls, *, max_iter: int = 200, tol: float = 0.01) -> Schedule:
        """Return the doc-04 ramp: p 1->3, then beta doubling 1->32."""
        pairs = [(1, 1), (2, 1), (3, 1), (3, 2), (3, 4), (3, 8), (3, 16), (3, 32)]
        return cls(tuple(Stage(float(p), float(b), max_iter, tol) for p, b in pairs))

    @classmethod
    def single(
        cls, *, p: float = 3.0, beta: float = 1.0, max_iter: int = 200, tol: float = 0.01
    ) -> Schedule:
        """One stage = no continuation (the 1.9a behavior)."""
        return cls((Stage(p, beta, max_iter, tol),))


@dataclass
class IterationState:
    """The per-iteration state yielded by :meth:`Study.iterate`."""

    iteration: int
    x: _F64
    objective: float
    change: float
    kkt: float
    stage: int


@dataclass
class Result:
    """The outcome of a study run."""

    design: DesignField
    x: _F64
    objective: float
    history: dict[str, list[float]]
    iterations: int
    converged: bool
    reason: str
    timing: float
    kkt: float


@dataclass
class Study:
    """Runs the optimization loop for a :class:`Problem`."""

    problem: Problem
    schedule: Schedule | None = None
    max_iter: int = 200
    tol: float = 0.01
    x0: _F64 | None = None
    snapshot_every: int = 5
    events: EventBus = field(default_factory=EventBus)

    def __post_init__(self) -> None:
        if self.max_iter < 1:
            raise ProblemError(f"max_iter must be >= 1, got {self.max_iter}")
        if self.schedule is None:
            # continuation ON by default (E7); max_iter/tol feed the per-stage caps
            self.schedule = Schedule.default(max_iter=self.max_iter, tol=self.tol)
        if self.x0 is not None:
            x0 = np.asarray(self.x0, dtype=np.float64)
            if x0.shape != (self.problem.chain.n_vars,):
                raise ProblemError(f"x0 shape {x0.shape} != ({self.problem.chain.n_vars},)")

    def _initial_x(self) -> _F64:
        if self.x0 is not None:
            x0 = np.asarray(self.x0, dtype=np.float64)
            if x0.shape != (self.problem.chain.n_vars,):
                raise ProblemError(f"x0 shape {x0.shape} != ({self.problem.chain.n_vars},)")
            return x0
        return self.problem.chain.initial_design(self.problem.default_volume_fraction())

    def _evaluate(
        self, x: _F64, chain: BoundChain
    ) -> tuple[Solution, float, _F64, _F64, _F64, dict[str, float]]:
        """Solve the physics at ``x`` through ``chain`` and return values+grads."""
        p = self.problem
        scale = chain.apply(x)
        rho = chain.physical_density(x)
        p.solver.prepare(p.model.assemble(scale))
        u = np.atleast_2d(np.asarray(p.solver.solve(p.model.loads()))).reshape(p.model.n_dof, -1)
        sol = Solution(
            model=p.model, mesh=p.model.mesh, displacements=u, interpolated=scale, density=rho
        )

        def grad_to_x(thing: Response | Constraint) -> _F64:
            gf = thing.grad_field(sol)
            if thing.field_basis == "interpolated":
                return np.asarray(chain.pullback(x, gf))
            return np.asarray(chain.pullback_density(x, gf))

        f0 = p.objective.value(sol)
        df0 = grad_to_x(p.objective)
        responses = {p.objective.name: f0}
        gvals = np.empty(len(p.constraints))
        dgs = np.empty((len(p.constraints), x.size))
        for i, c in enumerate(p.constraints):
            gvals[i] = c.value(sol)
            dgs[i] = grad_to_x(c)
            responses[c.report_key] = c.response.value(sol)
        return sol, f0, df0, gvals, dgs, responses

    def run(self) -> Result:
        """Run every continuation stage to convergence or the cap, firing events."""
        for _ in self.iterate():
            pass
        x = self._final.x  # the evaluated design (paired with self._final.objective)
        return Result(
            design=DesignField(
                self._final_chain.physical_density(x), self.problem.model.mesh, name="density"
            ),
            x=x,
            objective=self._final.objective,
            history=self._history,
            iterations=self._final.iteration,
            converged=self._converged,
            reason=self._reason,
            timing=self._timing,
            kkt=self._final.kkt,
        )

    def iterate(self) -> Iterator[IterationState]:
        """Yield one :class:`IterationState` per iteration across all stages.

        Each yielded state pairs the *evaluated* design ``x`` with the
        objective measured at that ``x``. Stages override the chain's SIMP
        ``p``/Heaviside ``beta`` (Option A); an executed stage warm-starts from
        the previous stage's ``x`` but resets the optimizer and ``c0``.
        """
        p = self.problem
        assert self.schedule is not None
        x = self._initial_x()
        self._history: dict[str, list[float]] = {
            "objective": [],
            "change": [],
            "kkt": [],
            "stage": [],
        }
        for c in p.constraints:
            self._history[c.report_key] = []
        self._converged = False
        self._reason = "max iterations reached"
        t0 = time.perf_counter()
        self.events.publish(StudyStarted(config={"stages": len(self.schedule.stages)}))

        iteration = 0
        prev_spec: Chain | None = None
        for stage_index, stage in enumerate(self.schedule.stages):
            chain = _staged_chain(p.chain.spec, p.model.mesh, p=stage.p, beta=stage.beta)
            if chain.spec == prev_spec:
                continue  # dedup: identical to the previous executed stage
            prev_spec = chain.spec
            self._final_chain = chain
            p.optimizer.setup(chain.n_vars, np.zeros(chain.n_vars), np.ones(chain.n_vars))
            c0: float | None = None
            self._converged = False
            stage_reason = "stage max iterations reached"

            for _ in range(1, stage.max_iter + 1):
                iteration += 1
                sol, f0, df0, gvals, dgs, responses = self._evaluate(x, chain)
                if c0 is None:
                    c0 = abs(f0) if abs(f0) > 1e-30 else 1.0  # normalization scale (MMA)
                step = p.optimizer.step(x, f0 / c0, df0 / c0, gvals, dgs)

                self._history["objective"].append(f0)
                self._history["change"].append(step.change)
                self._history["kkt"].append(step.kkt)
                self._history["stage"].append(float(stage_index))
                for name, val in responses.items():
                    if name != p.objective.name:
                        self._history[name].append(val)
                self.events.publish(
                    IterationFinished(
                        iteration=iteration,
                        design_change=step.change,
                        responses=responses,
                        wall_time=time.perf_counter() - t0,
                        kkt=step.kkt,
                        stage=stage_index,
                    )
                )
                if self.snapshot_every and iteration % self.snapshot_every == 0:
                    self.events.publish(FieldSnapshot(iteration=iteration, rho=sol.density))

                self._final = IterationState(
                    iteration=iteration,
                    x=x.copy(),
                    objective=f0,
                    change=step.change,
                    kkt=step.kkt,
                    stage=stage_index,
                )
                yield self._final

                if step.change < stage.tol:
                    self._converged = True
                    stage_reason = f"design change {step.change:.2e} < tol {stage.tol:g}"
                    break
                x = step.x_next

            self._reason = stage_reason
            self.events.publish(StageFinished(stage=stage_index, reason=stage_reason))

        self._timing = time.perf_counter() - t0
        self.events.publish(
            StudyFinished(
                reason=self._reason,
                summary={"iterations": iteration, "converged": self._converged},
            )
        )
