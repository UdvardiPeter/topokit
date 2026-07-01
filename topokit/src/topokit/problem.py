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

import os
import time
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt

from topokit.checkpoint import SCHEMA_VERSION, config_fingerprint, read_topo, write_topo
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


def _staged_spec(spec: Chain, *, p: float, beta: float) -> Chain:
    """Return ``spec`` with SIMP ``p`` and Heaviside ``beta`` replaced (no bind).

    Continuation overrides the authored ``SIMP.p``/``Heaviside.beta`` with the
    stage values; other link types pass through. ``replace`` re-runs each
    link's ``__post_init__``, so a bad staged value fails loud before the stage.
    Kept separate from the bind so the stage-dedup compares specs cheaply.
    """
    links = tuple(
        replace(link, p=p)
        if isinstance(link, SIMP)
        else replace(link, beta=beta)
        if isinstance(link, Heaviside)
        else link
        for link in spec.links
    )
    return Chain(links)


def _staged_chain(spec: Chain, mesh: StructuredGrid, *, p: float, beta: float) -> BoundChain:
    """Rebind ``spec`` with the stage's SIMP ``p``/Heaviside ``beta`` (Option A)."""
    return _staged_spec(spec, p=p, beta=beta).bind(mesh)


_NODES_PER_ELEMENT = {"quad4": 4, "hex8": 8}


def _estimate_stiffness_bytes(n_dof: int, nnz: int) -> int:
    """Rough CSR float64 stiffness-matrix size: data + indices + indptr."""
    return nnz * (8 + 4) + (n_dof + 1) * 4


def _available_ram_bytes() -> int | None:
    """Best-effort total RAM in bytes, or None if undetectable."""
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def _warn_if_memory_tight(model: PhysicsModel) -> None:
    """Warn (with numbers) if the estimated stiffness matrix may strain RAM."""
    nen = _NODES_PER_ELEMENT.get(model.mesh.element_kind)
    if nen is None:
        return
    nnz = model.mesh.n_elements * (nen * model.mesh.dim) ** 2
    estimate = _estimate_stiffness_bytes(model.n_dof, nnz)
    ram = _available_ram_bytes()
    if ram is not None and estimate > 0.5 * ram:
        warnings.warn(
            f"estimated stiffness matrix ~{estimate / 1e9:.1f} GB may strain available "
            f"RAM ~{ram / 1e9:.1f} GB; consider a coarser mesh or the AmgCG solver",
            stacklevel=3,
        )


def _schedule_to_json(schedule: Schedule) -> list[dict[str, float]]:
    return [
        {"p": s.p, "beta": s.beta, "max_iter": s.max_iter, "tol": s.tol} for s in schedule.stages
    ]


def _schedule_from_json(data: list[dict[str, float]]) -> Schedule:
    return Schedule(
        tuple(
            Stage(p=d["p"], beta=d["beta"], max_iter=int(d["max_iter"]), tol=d["tol"]) for d in data
        )
    )


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
        _warn_if_memory_tight(model)

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
class _Resume:
    """The restored cursor for :meth:`Study.resume` (internal)."""

    stage_index: int
    iter_in_stage: int
    iteration: int
    x: _F64
    c0: float
    opt_state: dict[str, object]
    history: dict[str, list[float]]
    best_x: _F64
    best_obj: float
    best_g: float
    best_feasible: bool
    converged: bool
    reason: str


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
    best_design: DesignField
    best_x: _F64
    best_objective: float
    history: dict[str, list[float]]
    iterations: int
    stages_run: int
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
    checkpoint_path: str | None = None
    checkpoint_every: int = 10
    events: EventBus = field(default_factory=EventBus)

    def __post_init__(self) -> None:
        self._resume: _Resume | None = None
        if self.max_iter < 1:
            raise ProblemError(f"max_iter must be >= 1, got {self.max_iter}")
        if self.checkpoint_path is not None:
            parent = os.path.dirname(self.checkpoint_path)
            if parent and not os.path.isdir(parent):
                raise ProblemError(f"checkpoint directory does not exist: {parent!r}")
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

    @classmethod
    def resume(cls, problem: Problem, path: str, *, events: EventBus | None = None) -> Study:
        """Reconstruct a Study from a ``.topo`` and continue; ``problem`` must match."""
        manifest, arrays = read_topo(path)
        if int(manifest.get("schema", -1)) != SCHEMA_VERSION:
            raise ProblemError(
                f"unsupported .topo schema {manifest.get('schema')!r}; expected {SCHEMA_VERSION}"
            )
        schedule = _schedule_from_json(manifest["schedule"])
        study = cls(problem, schedule=schedule)
        if events is not None:
            study.events = events
        if config_fingerprint(study._config_parts()) != manifest["fingerprint"]:
            raise ProblemError("checkpoint was written for a different problem")
        opt_state: dict[str, object] = dict(manifest["opt_scalars"])
        for name in manifest["opt_none"]:
            opt_state[name] = None
        for key, arr in arrays.items():
            if key.startswith("opt__"):
                opt_state[key[len("opt__") :]] = arr
        study._resume = _Resume(
            stage_index=int(manifest["stage_index"]),
            iter_in_stage=int(manifest["iter_in_stage"]),
            iteration=int(manifest["iteration"]),
            x=arrays["x"],
            c0=float(manifest["c0"]),
            opt_state=opt_state,
            history={k: list(v) for k, v in manifest["history"].items()},
            best_x=arrays["best_x"],
            best_obj=float(manifest["best_objective"]),
            best_g=float(manifest["best_g"]),
            best_feasible=bool(manifest["best_feasible"]),
            converged=bool(manifest["converged"]),
            reason=str(manifest["reason"]),
        )
        return study

    def _config_parts(self) -> tuple[object, ...]:
        """Canonical view of the problem's structure for the checkpoint fingerprint."""
        p = self.problem
        assert self.schedule is not None
        return (
            tuple(p.model.mesh.shape),
            tuple(float(s) for s in p.model.mesh.spacing),
            int(p.model.n_dof),
            repr(p.chain.spec),
            p.objective.name,
            tuple((c.report_key, float(c.bound), c.sense) for c in p.constraints),
            repr(p.optimizer),  # dataclass repr carries the hyperparameters
            tuple(tuple(sorted(d.items())) for d in _schedule_to_json(self.schedule)),
        )

    def _checkpoint(
        self, stage_index: int, iter_in_stage: int, iteration: int, x: _F64, c0: float | None
    ) -> None:
        """Write the resumable state to ``checkpoint_path`` (atomic .topo)."""
        if self.checkpoint_path is None:
            return
        assert self.schedule is not None
        opt_state = self.problem.optimizer.state()
        opt_scalars: dict[str, object] = {}
        opt_arrays: dict[str, _F64] = {}
        opt_none: list[str] = []
        for k, v in opt_state.items():
            if v is None:
                opt_none.append(k)
            elif isinstance(v, np.ndarray):
                opt_arrays[f"opt__{k}"] = v
            else:
                opt_scalars[k] = v
        manifest = {
            "schema": SCHEMA_VERSION,
            "fingerprint": config_fingerprint(self._config_parts()),
            "schedule": _schedule_to_json(self.schedule),
            "stage_index": stage_index,
            "iter_in_stage": iter_in_stage,
            "iteration": iteration,
            "c0": c0,
            "converged": self._converged,
            "reason": self._reason,
            "history": self._history,
            "best_objective": self._best_obj,
            "best_g": self._best_g,
            "best_feasible": self._best_feasible,
            "opt_scalars": opt_scalars,
            "opt_none": opt_none,
        }
        arrays = {"x": np.asarray(x), "best_x": np.asarray(self._best_x), **opt_arrays}
        write_topo(self.checkpoint_path, manifest, arrays)

    def run(self) -> Result:
        """Run every continuation stage to convergence or the cap, firing events."""
        for _ in self.iterate():
            pass
        x = self._final.x  # the evaluated design (paired with self._final.objective)
        mesh = self.problem.model.mesh
        return Result(
            design=DesignField(self._final_chain.physical_density(x), mesh, name="density"),
            x=x,
            objective=self._final.objective,
            best_design=DesignField(
                self._final_chain.physical_density(self._best_x), mesh, name="density"
            ),
            best_x=self._best_x,
            best_objective=self._best_obj,
            history=self._history,
            iterations=self._final.iteration,
            stages_run=len({int(s) for s in self._history["stage"]}),
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
        resume = self._resume
        self._resume = None  # consume; a re-run starts fresh
        if resume is None:
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
            iteration = 0
            start_stage, start_j = 0, 1
        else:
            x = resume.x
            self._history = resume.history
            self._converged = resume.converged
            self._reason = resume.reason
            iteration = resume.iteration
            start_stage, start_j = resume.stage_index, resume.iter_in_stage
        t0 = time.perf_counter()
        self.events.publish(StudyStarted(config={"stages": len(self.schedule.stages)}))

        ran_any = False
        prev_spec: Chain | None = None
        for stage_index, stage in enumerate(self.schedule.stages):
            if stage_index < start_stage:
                continue  # already completed in a prior session (resume)
            staged = _staged_spec(p.chain.spec, p=stage.p, beta=stage.beta)
            if staged == prev_spec:
                continue  # dedup: identical to the previous executed stage (no bind paid)
            prev_spec = staged
            chain = staged.bind(p.model.mesh)
            self._final_chain = chain
            p.optimizer.setup(chain.n_vars, np.zeros(chain.n_vars), np.ones(chain.n_vars))
            c0: float | None = None
            stage_reason = "stage max iterations reached"
            if resume is not None and stage_index == start_stage:
                # restore the in-progress stage exactly (determinism: x, optimizer
                # state, c0, and best are all restored)
                p.optimizer.load_state(resume.opt_state)
                c0 = resume.c0
                self._best_feasible = resume.best_feasible
                self._best_x = resume.best_x
                self._best_obj = resume.best_obj
                self._best_g = resume.best_g
                j_start = start_j
            else:
                # best-feasible tracking, reset per stage: compliance across stages
                # is not comparable (different p), and early grey stages are not
                # valid topologies, so "best" means the best iterate of this stage.
                self._best_feasible = False
                self._best_x = x.copy()
                self._best_obj = float("inf")
                self._best_g = float("inf")
                j_start = 1
            self._converged = False

            for j in range(j_start, stage.max_iter + 1):
                iteration += 1
                ran_any = True
                sol, f0, df0, gvals, dgs, responses = self._evaluate(x, chain)
                if c0 is None:
                    c0 = abs(f0) if abs(f0) > 1e-30 else 1.0  # normalization scale (MMA)
                step = p.optimizer.step(x, f0 / c0, df0 / c0, gvals, dgs)

                maxg = float(gvals.max()) if gvals.size else 0.0
                if maxg <= 1e-4 and (not self._best_feasible or f0 < self._best_obj):
                    self._best_feasible, self._best_x, self._best_obj, self._best_g = (
                        True,
                        x.copy(),
                        f0,
                        maxg,
                    )
                elif not self._best_feasible and maxg < self._best_g:
                    self._best_x, self._best_obj, self._best_g = x.copy(), f0, maxg

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
                    self.events.publish(
                        FieldSnapshot(iteration=iteration, rho=sol.density, mesh=sol.mesh)
                    )

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
                # checkpoint the next-iteration cursor (x is the design to resume from,
                # the optimizer state is post-step, c0 is this stage's scale);
                # checkpoint_every <= 0 means "only the final write" (no per-iteration)
                if (
                    self.checkpoint_path
                    and self.checkpoint_every > 0
                    and iteration % self.checkpoint_every == 0
                ):
                    self._checkpoint(stage_index, j + 1, iteration, x, c0)

            self._reason = stage_reason
            self.events.publish(StageFinished(stage=stage_index, reason=stage_reason))

        if not ran_any:
            raise ProblemError(
                "nothing to resume: the checkpoint is already at the end of its schedule; "
                "extend the schedule (e.g. a larger max_iter) to continue"
            )
        self._checkpoint(stage_index, j + 1, iteration, x, c0)
        self._timing = time.perf_counter() - t0
        self.events.publish(
            StudyFinished(
                reason=self._reason,
                summary={"iterations": iteration, "converged": self._converged},
            )
        )
