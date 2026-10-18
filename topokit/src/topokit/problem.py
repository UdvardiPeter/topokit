# SPDX-License-Identifier: MIT
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

import hashlib
import os
import time
from collections.abc import Generator
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

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
from topokit.exceptions import TopoKitError
from topokit.fem import PhysicsModel
from topokit.fields import DesignField
from topokit.mesh import Mesh
from topokit.optimizers import OC, Optimizer
from topokit.parametrization import SIMP, BoundChain, Chain, Heaviside, LinkSpec
from topokit.responses import Compliance, Constraint, Response, Solution
from topokit.solvers import LinearSolver, auto_solver

if TYPE_CHECKING:  # the viz return types, for callers who type-check
    from matplotlib.figure import Figure
    from pyvista import Plotter

_F64 = npt.NDArray[np.float64]


class ProblemError(TopoKitError, ValueError):
    """Invalid problem assembly or study configuration."""


def _conforms(obj: object, protocol: type[object], *methods: str) -> bool:
    """``isinstance(obj, protocol)`` plus an explicit callable check on ``methods``.

    ``isinstance`` against a runtime-checkable ``Protocol`` only checks that
    each member is present, not that a member meant to be a method is
    actually callable; a non-callable attribute of the same name (e.g. a
    hyperparameter that happens to be called ``step``) would otherwise shadow
    the real method and pass undetected until the orchestration loop calls it.
    A class object is rejected outright: its methods are present and callable,
    but the caller almost certainly forgot to instantiate it.
    """
    if isinstance(obj, type):
        return False
    return isinstance(obj, protocol) and all(callable(getattr(obj, m, None)) for m in methods)


def _component_label(obj: object) -> str:
    """Name a component for an error message, catching the class-for-instance slip."""
    if isinstance(obj, type):
        return f"{obj.__name__!r} (the class itself; did you mean {obj.__name__}()?)"
    return repr(type(obj).__name__)


_PHYSICS_METHODS = ("assemble", "loads", "element_energies", "element_stress")
_RESPONSE_METHODS = ("value", "grad_field")
_OPTIMIZER_METHODS = ("setup", "step", "state", "load_state")
_SOLVER_METHODS = ("prepare", "solve")


def _bind_chain(chain: LinkSpec | Chain | BoundChain, model: PhysicsModel) -> BoundChain:
    """Bind whatever the caller passed: a bound chain, a spec, or a lone link.

    ``LinkSpec.bind`` makes a one-link chain, so a terminal link on its own is
    a valid parametrization (SIMP with no filter, say). A non-terminal lone
    link still fails in ``Chain.bind`` with the terminal-link error.
    """
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


def _mesh_digest(mesh: Mesh) -> str:
    """Hex digest of everything about a mesh the design vector is indexed by.

    Node coordinates, connectivity and the solid/void masks. Counts alone
    cannot tell two meshes apart: re-splitting a quad along the other
    diagonal keeps every count and every coordinate while making element 0 a
    different triangle, moving a tet10 midside node leaves every centroid
    unchanged, and swapping which element is solid changes neither n_dof nor
    the design-variable count.
    """
    digest = hashlib.sha256()
    for part in (mesh.nodes, mesh.element_nodes, mesh.solid, mesh.void):
        digest.update(np.ascontiguousarray(part).tobytes())
    return digest.hexdigest()


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
        chain: LinkSpec | Chain | BoundChain,
        objective: Response,
        constraints: list[Constraint] | tuple[Constraint, ...] = (),
        optimizer: Optimizer | None = None,
        solver: LinearSolver | str = "auto",
    ) -> None:
        if not _conforms(model, PhysicsModel, *_PHYSICS_METHODS):
            raise ProblemError(
                f"model {_component_label(model)} does not satisfy the PhysicsModel protocol"
            )
        self.model = model
        self.chain = _bind_chain(chain, model)
        if self.chain.out_field != model.expected_field:
            raise ProblemError(
                f"chain output field {self.chain.out_field} does not match the physics "
                f"expected field {model.expected_field}"
            )
        self.objective = objective
        if not _conforms(objective, Response, *_RESPONSE_METHODS):
            raise ProblemError(
                f"objective {_component_label(objective)} does not satisfy the Response protocol"
            )
        self.constraints = tuple(constraints)
        # Events and history key responses by name; collisions would silently
        # drop a constraint from the reporting, so reject them up front. The
        # objective lives under "objective"; "change" is the step-size series.
        reserved = {"objective", "change", objective.name}
        seen: set[str] = set()
        for i, c in enumerate(self.constraints):
            if not isinstance(c, Constraint):
                # c is typed as Constraint (declared element type), but nothing
                # stops a caller from passing something else at runtime -- the
                # mistake this check exists to catch.
                hint = (  # type: ignore[unreachable]
                    "; bound it first, e.g. Volume() <= 0.4" if isinstance(c, Response) else ""
                )
                raise ProblemError(
                    f"constraints[{i}] is {type(c).__name__!r}, not a Constraint{hint}"
                )
            if not _conforms(c.response, Response, *_RESPONSE_METHODS):
                raise ProblemError(
                    f"constraints[{i}].response {_component_label(c.response)} does not "
                    "satisfy the Response protocol"
                )
            key = c.report_key
            if key in reserved or key in seen:
                raise ProblemError(
                    f"constraint report key {key!r} collides with the objective, another "
                    f"constraint, or a reserved history key; give the constraint a distinct "
                    f"label via Constraint.labeled(...)"
                )
            seen.add(key)
        self.optimizer = optimizer if optimizer is not None else OC()
        if not _conforms(self.optimizer, Optimizer, *_OPTIMIZER_METHODS):
            raise ProblemError(
                f"optimizer {_component_label(self.optimizer)} does not satisfy the Optimizer "
                "protocol (setup/step/state/load_state)"
            )
        # Fail-loud at construction, not after the first (possibly expensive)
        # FE solve: OC's single-constraint requirement and Compliance's
        # per-case weight count are both knowable here.
        if isinstance(self.optimizer, OC) and len(self.constraints) != 1:
            raise ProblemError(
                f"OC handles exactly one constraint, got {len(self.constraints)}; "
                "use MMA for general constraint sets"
            )
        for resp in (objective, *(c.response for c in self.constraints)):
            if (
                isinstance(resp, Compliance)
                and resp.weights is not None
                and len(resp.weights) != model.n_cases
            ):
                raise ProblemError(
                    f"{resp.name}: {len(resp.weights)} weights for {model.n_cases} load cases"
                )
        if isinstance(solver, str):
            if solver != "auto":
                raise ProblemError(f"unknown solver {solver!r}; use 'auto' or a LinearSolver")
            self.solver: LinearSolver = auto_solver(model.n_dof, model.mesh.dim)
        else:
            self.solver = solver
        if not _conforms(self.solver, LinearSolver, *_SOLVER_METHODS):
            raise ProblemError(
                f"solver {_component_label(self.solver)} does not satisfy the LinearSolver "
                "protocol (prepare/solve)"
            )
        # AMG solvers benefit from the model's rigid-body modes; wired by
        # capability so neither protocol has to grow (a solver or model
        # without the pairing is simply left alone).
        if hasattr(self.solver, "set_near_nullspace") and hasattr(model, "near_nullspace"):
            self.solver.set_near_nullspace(model.near_nullspace())

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
    per-stage convergence cap/tol (``200/stage``).
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
    """An ordered tuple of continuation stages (introspectable/replaceable)."""

    stages: tuple[Stage, ...]

    def __post_init__(self) -> None:
        if not self.stages:
            raise ProblemError("schedule stages must not be empty")

    @classmethod
    def default(cls, *, max_iter: int = 200, tol: float = 0.01) -> Schedule:
        """Return the default ramp: p 1->3, then beta doubling 1->16.

        The beta ramp stops at 16, amending the originally planned 32:
        measured across meshes, the beta=32 stage destabilized converged
        designs (near-total projection saturation kills the gradients MMA
        and OC steer by) while sharpening the design only marginally. Pass
        an explicit :class:`Schedule` to ramp further.
        """
        pairs = [(1, 1), (2, 1), (3, 1), (3, 2), (3, 4), (3, 8), (3, 16)]
        return cls(tuple(Stage(float(p), float(b), max_iter, tol) for p, b in pairs))

    @classmethod
    def single(
        cls, *, p: float = 3.0, beta: float = 1.0, max_iter: int = 200, tol: float = 0.01
    ) -> Schedule:
        """One stage = no continuation."""
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
class _RunState:
    """Everything a run accumulates, in one place.

    ``iterate`` used to hang these off ``self``, which meant ``run`` and
    ``_checkpoint`` read ten attributes that nothing in ``__init__`` created.
    """

    history: dict[str, list[float]]
    converged: bool = False
    reason: str = "max iterations reached"
    best_feasible: bool = False
    best_x: _F64 = field(default_factory=lambda: np.zeros(0))
    best_objective: float = float("inf")
    best_g: float = float("inf")
    timing: float = 0.0
    final: IterationState | None = None
    final_chain: BoundChain | None = None

    def offer(self, x: _F64, objective: float, max_g: float) -> None:
        """Keep ``x`` if it is the best iterate of this stage so far.

        Feasible beats infeasible; among feasible, lowest objective; among
        infeasible, least violation.
        """
        if max_g <= 1e-4 and (not self.best_feasible or objective < self.best_objective):
            self.best_feasible, self.best_x, self.best_objective, self.best_g = (
                True,
                x.copy(),
                objective,
                max_g,
            )
        elif not self.best_feasible and max_g < self.best_g:
            self.best_x, self.best_objective, self.best_g = x.copy(), objective, max_g

    def reset_best(self, x: _F64) -> None:
        """Forget the previous stage's best.

        Compliance is not comparable across stages (different ``p``) and early
        grey stages are not valid topologies, so "best" means the best iterate
        of the current stage.
        """
        self.best_feasible = False
        self.best_x = x.copy()
        self.best_objective = float("inf")
        self.best_g = float("inf")


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

    def view(self, **kwargs: Any) -> Figure | Plotter:
        """Render the final design (2D heatmap / 3D iso-surface). Needs ``[viz]``."""
        from topokit.viz import view

        return view(self, **kwargs)

    def plot_convergence(self, **kwargs: Any) -> Figure:
        """Plot objective/constraint/Δx curves. Needs ``[viz]``."""
        from topokit.viz import plot_convergence

        return plot_convergence(self, **kwargs)

    def view_slices(self, **kwargs: Any) -> Figure:
        """Render 3D density cross-sections. Needs ``[viz]``."""
        from topokit.viz import view_slices

        return view_slices(self, **kwargs)


DEFAULT_MAX_ITER = 200
DEFAULT_TOL = 0.01


@dataclass
class Study:
    """Runs the optimization loop for a :class:`Problem`.

    ``max_iter`` and ``tol`` are the per-stage cap and convergence tolerance
    of the default continuation schedule. A ``schedule`` carries its own, so
    passing both is rejected rather than silently resolved.
    """

    problem: Problem
    schedule: Schedule | None = None
    max_iter: int = DEFAULT_MAX_ITER
    tol: float = DEFAULT_TOL
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
            # continuation is on by default; max_iter/tol feed the per-stage caps
            self.schedule = Schedule.default(max_iter=self.max_iter, tol=self.tol)
        else:
            # Every stage carries its own cap and tol, so these would do
            # nothing: a Study(schedule=Schedule.single(max_iter=40),
            # max_iter=3) used to run 40 iterations without a word. Knowable
            # at construction, so it fails here like the rest of the assembly.
            shadowed = [
                f"{name}={value!r}"
                for name, value, default in (
                    ("max_iter", self.max_iter, DEFAULT_MAX_ITER),
                    ("tol", self.tol, DEFAULT_TOL),
                )
                if value != default
            ]
            if shadowed:
                raise ProblemError(
                    f"{' and '.join(shadowed)} would be ignored: the given schedule carries "
                    "a cap and tolerance per stage. Set them on the schedule instead, e.g. "
                    "Schedule.single(max_iter=..., tol=...) or Schedule.default(max_iter=...)"
                )
        if self.x0 is not None:
            x0 = np.asarray(self.x0, dtype=np.float64)
            if x0.shape != (self.problem.chain.n_vars,):
                raise ProblemError(f"x0 shape {x0.shape} != ({self.problem.chain.n_vars},)")

    def _initial_x(self) -> _F64:
        if self.x0 is not None:
            return np.asarray(self.x0, dtype=np.float64)  # shape checked in __post_init__
        return self.problem.chain.initial_design(self.problem.default_volume_fraction())

    def _evaluate(
        self, x: _F64, chain: BoundChain
    ) -> tuple[Solution, float, _F64, _F64, _F64, dict[str, float]]:
        """Solve the physics at ``x`` through ``chain`` and return values+grads."""
        p = self.problem
        ev = chain.evaluate(x)
        scale = ev.field
        rho = ev.density
        p.solver.prepare(p.model.assemble(scale))
        u = np.atleast_2d(np.asarray(p.solver.solve(p.model.loads()))).reshape(p.model.n_dof, -1)
        sol = Solution(
            model=p.model, mesh=p.model.mesh, displacements=u, interpolated=scale, density=rho
        )

        def grad_to_x(thing: Response | Constraint) -> _F64:
            gf = thing.grad_field(sol)
            if thing.field_basis == "interpolated":
                return np.asarray(ev.pullback(gf))
            return np.asarray(ev.pullback_density(gf))

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
    def resume(
        cls,
        problem: Problem,
        path: str,
        *,
        events: EventBus | None = None,
    ) -> Study:
        """Reconstruct a Study from a ``.topo`` and continue; ``problem`` must match.

        The run continues under the schedule the checkpoint was written with,
        from the iteration its last checkpoint recorded, and the problem's
        fingerprint is verified against the one stored in the file. To run
        further than that schedule allows, start a new study from the result:
        ``Study(problem, schedule=longer, x0=result.x)``.
        """
        manifest, arrays = read_topo(path)
        if int(manifest.get("schema", -1)) != SCHEMA_VERSION:
            raise ProblemError(
                f"unsupported .topo schema {manifest.get('schema')!r}; expected {SCHEMA_VERSION}"
            )
        written = _schedule_from_json(manifest["schedule"])
        study = cls(problem, schedule=written)
        if events is not None:
            study.events = events
        if config_fingerprint(study._config_parts()) != manifest["fingerprint"]:
            raise ProblemError("checkpoint was written for a different problem")
        stage_index = int(manifest["stage_index"])
        opt_state: dict[str, object] = dict(manifest["opt_scalars"])
        for name in manifest["opt_none"]:
            opt_state[name] = None
        for key, arr in arrays.items():
            if key.startswith("opt__"):
                opt_state[key[len("opt__") :]] = arr
        study._resume = _Resume(
            stage_index=stage_index,
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
        mesh = p.model.mesh
        # repr()-hashed into the checkpoint fingerprint. One digest covers every
        # mesh kind: the grid's shape and spacing are implied by its nodes, and
        # SCHEMA_VERSION 3 announces that the layout changed.
        return (
            _mesh_digest(mesh),
            int(p.model.n_dof),
            repr(p.chain.spec),
            p.objective.name,
            tuple((c.report_key, float(c.bound), c.sense) for c in p.constraints),
            repr(p.optimizer),  # dataclass repr carries the hyperparameters
            tuple(tuple(sorted(d.items())) for d in _schedule_to_json(self.schedule)),
        )

    def _checkpoint(
        self,
        state: _RunState,
        stage_index: int,
        iter_in_stage: int,
        iteration: int,
        x: _F64,
        c0: float | None,
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
            "converged": state.converged,
            "reason": state.reason,
            "history": state.history,
            "best_objective": state.best_objective,
            "best_g": state.best_g,
            "best_feasible": state.best_feasible,
            "opt_scalars": opt_scalars,
            "opt_none": opt_none,
        }
        arrays = {"x": np.asarray(x), "best_x": np.asarray(state.best_x), **opt_arrays}
        write_topo(self.checkpoint_path, manifest, arrays)

    def run(self) -> Result:
        """Run every continuation stage to convergence or the cap, firing events."""
        gen = self.iterate()
        while True:
            try:
                next(gen)
            except StopIteration as stop:
                final: Result = stop.value
                return final

    def _result(self, state: _RunState, iterations: int) -> Result:
        """Package a finished run."""
        assert state.final is not None and state.final_chain is not None
        mesh = self.problem.model.mesh
        x = state.final.x  # the evaluated design, paired with state.final.objective
        return Result(
            design=DesignField(state.final_chain.physical_density(x), mesh, name="density"),
            x=x,
            objective=state.final.objective,
            best_design=DesignField(
                state.final_chain.physical_density(state.best_x), mesh, name="density"
            ),
            best_x=state.best_x,
            best_objective=state.best_objective,
            history=state.history,
            iterations=iterations,
            stages_run=len({int(v) for v in state.history["stage"]}),
            converged=state.converged,
            reason=state.reason,
            timing=state.timing,
            kkt=state.final.kkt,
        )

    def iterate(self) -> Generator[IterationState, None, Result]:
        """Yield one :class:`IterationState` per iteration across all stages.

        Returns the :class:`Result` when the generator finishes, which is
        what :meth:`run` collects.

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
            history: dict[str, list[float]] = {
                "objective": [],
                "change": [],
                "kkt": [],
                "stage": [],
            }
            for c in p.constraints:
                history[c.report_key] = []
            state = _RunState(history=history)
            iteration = 0
            start_stage, start_j = 0, 1
        else:
            x = resume.x
            state = _RunState(
                history=resume.history, converged=resume.converged, reason=resume.reason
            )
            iteration = resume.iteration
            start_stage, start_j = resume.stage_index, resume.iter_in_stage
        t0 = time.perf_counter()
        self.events.publish(StudyStarted(config={"stages": len(self.schedule.stages)}))

        # Decide which stages run before running any: a stage whose staged spec
        # repeats the previous one changes nothing, and resolving that here
        # means no skipped stage index can leak into a checkpoint cursor.
        planned: list[tuple[int, Stage, Chain]] = []
        for index, stage in enumerate(self.schedule.stages):
            if index < start_stage:
                continue  # already completed in a prior session (resume)
            staged = _staged_spec(p.chain.spec, p=stage.p, beta=stage.beta)
            if planned and staged == planned[-1][2]:
                continue  # identical to the previous executed stage: no bind paid
            planned.append((index, stage, staged))

        ran_any = False
        final_cursor: tuple[int, int, float | None] | None = None
        for stage_index, stage, staged in planned:
            chain = staged.bind(p.model.mesh)
            state.final_chain = chain
            p.optimizer.setup(chain.n_vars, np.zeros(chain.n_vars), np.ones(chain.n_vars))
            c0: float | None = None
            stage_reason = "stage max iterations reached"
            if resume is not None and stage_index == start_stage:
                # restore the in-progress stage exactly (determinism: x, optimizer
                # state, c0, and best are all restored)
                p.optimizer.load_state(resume.opt_state)
                c0 = resume.c0
                state.best_feasible = resume.best_feasible
                state.best_x = resume.best_x
                state.best_objective = resume.best_obj
                state.best_g = resume.best_g
                j_start = start_j
            else:
                state.reset_best(x)
                j_start = 1
            state.converged = False

            reached = 0  # the last iteration this stage actually ran
            for j in range(j_start, stage.max_iter + 1):
                iteration += 1
                reached = j
                ran_any = True
                sol, f0, df0, gvals, dgs, responses = self._evaluate(x, chain)
                if c0 is None:
                    c0 = abs(f0) if abs(f0) > 1e-30 else 1.0  # normalization scale (MMA)
                step = p.optimizer.step(x, f0 / c0, df0 / c0, gvals, dgs)

                maxg = float(gvals.max()) if gvals.size else 0.0
                state.offer(x, f0, maxg)

                state.history["objective"].append(f0)
                state.history["change"].append(step.change)
                state.history["kkt"].append(step.kkt)
                state.history["stage"].append(float(stage_index))
                for name, val in responses.items():
                    if name != p.objective.name:
                        state.history[name].append(val)
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

                state.final = IterationState(
                    iteration=iteration,
                    x=x.copy(),
                    objective=f0,
                    change=step.change,
                    kkt=step.kkt,
                    stage=stage_index,
                )
                yield state.final

                if step.change < stage.tol:
                    state.converged = True
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
                    self._checkpoint(state, stage_index, j + 1, iteration, x, c0)

            state.reason = stage_reason
            if reached:
                # resuming at the end of a stage leaves the range empty, so only
                # a stage that ran can name the cursor to resume from
                final_cursor = (stage_index, reached + 1, c0)
            self.events.publish(StageFinished(stage=stage_index, reason=stage_reason))

        if not ran_any or final_cursor is None:
            raise ProblemError(
                "nothing to resume: the checkpoint is already at the end of its schedule. "
                "Start a new study from the result instead: "
                "Study(problem, schedule=longer, x0=result.x)"
            )
        self._checkpoint(state, *final_cursor[:2], iteration, x, final_cursor[2])
        state.timing = time.perf_counter() - t0
        self.events.publish(
            StudyFinished(
                reason=state.reason,
                summary={"iterations": iteration, "converged": state.converged},
            )
        )
        return self._result(state, iteration)
