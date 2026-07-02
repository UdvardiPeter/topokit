# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Tests for the clean-room MMA optimizer and its subproblem solver."""

from typing import Any

import numpy as np
import pytest
import scipy.optimize

from topokit.optimizers import MMA, MMASubproblem, OptimizerError, solve_subproblem


def _random_subproblem(n: int, m: int, seed: int) -> MMASubproblem:
    rng = np.random.default_rng(seed)
    lower = rng.uniform(-2.0, -1.0, n)
    upper = rng.uniform(1.0, 2.0, n)
    alpha = lower + 0.1 * (upper - lower)
    beta = upper - 0.1 * (upper - lower)
    p0 = rng.uniform(0.5, 2.0, n)
    q0 = rng.uniform(0.5, 2.0, n)
    p = rng.uniform(0.1, 1.0, (m, n))
    q = rng.uniform(0.1, 1.0, (m, n))
    # choose b so the midpoint is strictly feasible (the subproblem is solvable
    # in x; otherwise y absorbs the violation and there is nothing for scipy to
    # match)
    x_mid = 0.5 * (alpha + beta)
    g_mid = np.array([(p[i] / (upper - x_mid) + q[i] / (x_mid - lower)).sum() for i in range(m)])
    b = g_mid + 0.5
    return MMASubproblem(
        low=lower,
        upp=upper,
        alpha=alpha,
        beta=beta,
        p0=p0,
        q0=q0,
        p=p,
        q=q,
        b=b,
        a0=1.0,
        a=np.zeros(m),
        c=np.full(m, 1000.0),
        d=np.ones(m),
    )


def _subproblem_objective(sp: MMASubproblem, x: np.ndarray) -> float:
    return float((sp.p0 / (sp.upp - x) + sp.q0 / (x - sp.low)).sum())


def _subproblem_constraints(sp: MMASubproblem, x: np.ndarray) -> np.ndarray:
    return np.array(
        [(sp.p[i] / (sp.upp - x) + sp.q[i] / (x - sp.low)).sum() - sp.b[i] for i in range(sp.m)]
    )


@pytest.mark.parametrize("m", [1, 3])
def test_subsolv_kkt_residual_small(m: int) -> None:
    sp = _random_subproblem(n=6, m=m, seed=m)
    sol = solve_subproblem(sp)
    # primal feasibility: x in [alpha, beta]; constraints satisfied (with y, z ~ 0)
    assert np.all(sol.x >= sp.alpha - 1e-7)
    assert np.all(sol.x <= sp.beta + 1e-7)
    assert np.all(_subproblem_constraints(sp, sol.x) <= 1e-5)


def test_subsolv_matches_scipy() -> None:
    sp = _random_subproblem(n=5, m=2, seed=11)
    sol = solve_subproblem(sp)

    def obj(x: np.ndarray) -> float:
        return _subproblem_objective(sp, x)

    cons = [
        {"type": "ineq", "fun": (lambda x, i=i: -_subproblem_constraints(sp, x)[i])}
        for i in range(sp.m)
    ]
    ref = scipy.optimize.minimize(  # type: ignore[call-overload]
        obj,
        0.5 * (sp.alpha + sp.beta),
        method="SLSQP",
        bounds=list(zip(sp.alpha, sp.beta, strict=True)),
        constraints=cons,
        options={"ftol": 1e-12, "maxiter": 500},
    )
    np.testing.assert_allclose(sol.x, ref.x, atol=1e-4)


def test_subsolv_1d_analytic() -> None:
    # min p/(U-x) + q/(x-L), no constraints; stationary at
    # x* = (U*sqrt(q) + L*sqrt(p)) / (sqrt(p) + sqrt(q))
    p0, q0, low, upp = 2.0, 0.5, 0.0, 4.0
    alpha, beta = 0.4, 3.6
    sp = MMASubproblem(
        low=np.array([low]),
        upp=np.array([upp]),
        alpha=np.array([alpha]),
        beta=np.array([beta]),
        p0=np.array([p0]),
        q0=np.array([q0]),
        p=np.zeros((0, 1)),
        q=np.zeros((0, 1)),
        b=np.zeros(0),
        a0=1.0,
        a=np.zeros(0),
        c=np.zeros(0),
        d=np.zeros(0),
    )
    sol = solve_subproblem(sp)
    xstar = (upp * np.sqrt(q0) + low * np.sqrt(p0)) / (np.sqrt(p0) + np.sqrt(q0))
    np.testing.assert_allclose(sol.x[0], xstar, atol=1e-5)


# ---- MMA end-to-end ----


def _mma(n: int) -> MMA:
    opt = MMA()
    opt.setup(n_vars=n, lower=np.zeros(n), upper=np.ones(n))
    return opt


def test_mma_matches_scipy_on_convex_problem() -> None:
    # min sum (x - t)^2  s.t.  mean(x) <= V
    n, V = 10, 0.3
    rng = np.random.default_rng(5)
    t = rng.uniform(0.0, 1.0, n)
    opt = _mma(n)
    x = np.full(n, V)
    for _ in range(150):
        f0 = float(((x - t) ** 2).sum())
        df0 = 2.0 * (x - t)
        g = np.array([x.mean() / V - 1.0])
        dg = np.full((1, n), (1.0 / n) / V)
        res = opt.step(x, f0, df0, g, dg)
        if res.change < 1e-7:
            x = res.x_next
            break
        x = res.x_next

    def obj(z: np.ndarray) -> float:
        return float(((z - t) ** 2).sum())

    ref = scipy.optimize.minimize(  # type: ignore[call-overload]
        obj,
        np.full(n, V),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "ineq", "fun": lambda z: V - z.mean()}],
        options={"ftol": 1e-12, "maxiter": 300},
    )
    np.testing.assert_allclose(x, ref.x, atol=1e-3)


def test_mma_two_constraints_matches_scipy() -> None:
    n = 6
    rng = np.random.default_rng(8)
    t = rng.uniform(0.0, 1.0, n)
    a1 = rng.uniform(0.1, 1.0, n)
    a2 = rng.uniform(0.1, 1.0, n)
    b1, b2 = 0.4 * a1.sum(), 0.5 * a2.sum()
    opt = _mma(n)
    x = np.full(n, 0.4)
    for _ in range(200):
        f0 = float(((x - t) ** 2).sum())
        df0 = 2.0 * (x - t)
        g = np.array([a1 @ x / b1 - 1.0, a2 @ x / b2 - 1.0])
        dg = np.vstack([a1 / b1, a2 / b2])
        res = opt.step(x, f0, df0, g, dg)
        if res.change < 1e-7:
            x = res.x_next
            break
        x = res.x_next
    ref = scipy.optimize.minimize(  # type: ignore[call-overload]
        lambda z: float(((z - t) ** 2).sum()),
        np.full(n, 0.4),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[
            {"type": "ineq", "fun": lambda z: b1 - a1 @ z},
            {"type": "ineq", "fun": lambda z: b2 - a2 @ z},
        ],
        options={"ftol": 1e-12, "maxiter": 300},
    )
    np.testing.assert_allclose(x, ref.x, atol=2e-3)


def test_mma_asymptote_band_stays_positive_under_monotone_march() -> None:
    # a steady monotone design march keeps the asymptote band open (gamma>=1),
    # exercising the k>=3 update; oscillation (handled by gamma=asydecr) shrinks
    # it, which the convergence test indirectly covers.
    opt = _mma(1)
    x = np.array([0.5])
    widths = []
    for v in (0.6, 0.7, 0.8, 0.9):
        opt.step(x, 0.0, np.array([-1.0]), np.array([-10.0]), np.array([[0.0]]))
        widths.append(float(opt.state()["upp"][0] - opt.state()["low"][0]))
        x = np.array([v])
    assert all(w > 0.0 for w in widths)
    assert widths[-1] >= widths[1] * 0.99  # not collapsing under a monotone march


def test_mma_subproblem_infeasible_relaxation() -> None:
    # when the subproblem is infeasible in x, y absorbs the violation and the
    # KKT stationarity gives lam ~ c + d*y (the path the early sign bug broke)
    n, m = 6, 1
    rng = np.random.default_rng(0)
    low = rng.uniform(-2.0, -1.0, n)
    upp = rng.uniform(1.0, 2.0, n)
    alpha = low + 0.1 * (upp - low)
    beta = upp - 0.1 * (upp - low)
    xm = 0.5 * (alpha + beta)
    p = rng.uniform(0.1, 1.0, (m, n))
    q = rng.uniform(0.1, 1.0, (m, n))
    g_mid = np.array([(p[0] / (upp - xm) + q[0] / (xm - low)).sum()])
    sp = MMASubproblem(
        low=low,
        upp=upp,
        alpha=alpha,
        beta=beta,
        p0=rng.uniform(0.5, 2.0, n),
        q0=rng.uniform(0.5, 2.0, n),
        p=p,
        q=q,
        b=g_mid - 5.0,  # unreachable -> infeasible in x
        a0=1.0,
        a=np.zeros(m),
        c=np.full(m, 1000.0),
        d=np.ones(m),
    )
    sol = solve_subproblem(sp)
    assert sol.y[0] > 0.1  # y takes up the violation
    np.testing.assert_allclose(sol.lam[0], 1000.0 + sol.y[0], rtol=1e-3)


def test_mma_negative_gradient_constraint() -> None:
    # a ">=" style constraint has a negative gradient (the q-branch); MMA must
    # still satisfy it. Forward validation for v1.x stress/displacement limits.
    n = 20
    rng = np.random.default_rng(2)
    t = rng.uniform(0.0, 1.0, n)
    opt = _mma(n)
    x = np.full(n, 0.5)
    for _ in range(120):
        df0 = 2.0 * (x - t)
        g = np.array([0.6 - x.mean()])  # mean(x) >= 0.6
        dg = np.full((1, n), -1.0 / n)
        res = opt.step(x, float(((x - t) ** 2).sum()), df0, g, dg)
        x = res.x_next
        if res.change < 1e-6:
            break
    assert x.mean() >= 0.6 - 1e-3


def test_mma_converges_in_bounded_iterations() -> None:
    n, V = 30, 0.3
    rng = np.random.default_rng(1)
    t = rng.uniform(0.0, 1.0, n)
    opt = _mma(n)
    x = np.full(n, V)
    iters = 0
    for it in range(1, 101):
        iters = it
        df0 = 2.0 * (x - t)
        g = np.array([x.mean() / V - 1.0])
        dg = np.full((1, n), (1.0 / n) / V)
        res = opt.step(x, float(((x - t) ** 2).sum()), df0, g, dg)
        x = res.x_next
        if res.change < 1e-6:
            break
    assert iters < 60  # MMA converges briskly (no oscillation stall)


def test_mma_state_roundtrip_identical_trajectory() -> None:
    n = 8
    rng = np.random.default_rng(3)
    t = rng.uniform(0.0, 1.0, n)

    def run(restore_at: int | None) -> np.ndarray:
        opt = _mma(n)
        x = np.full(n, 0.3)
        for k in range(20):
            if restore_at is not None and k == restore_at:
                saved = opt.state()
                opt2 = _mma(n)
                opt2.load_state(saved)
                opt = opt2
            df0 = 2.0 * (x - t)
            g = np.array([x.mean() / 0.3 - 1.0])
            dg = np.full((1, n), (1.0 / n) / 0.3)
            x = opt.step(x, float(((x - t) ** 2).sum()), df0, g, dg).x_next
        return x

    np.testing.assert_array_equal(run(None), run(10))


def test_mma_matches_oc_on_cantilever() -> None:
    from topokit.fem import LinearElasticity, Material, PointLoad
    from topokit.mesh import StructuredGrid
    from topokit.optimizers import OC
    from topokit.parametrization import SIMP, DensityFilter
    from topokit.responses import Compliance, Solution, Volume
    from topokit.selection import Box, PlaneSlab
    from topokit.solvers import Direct

    def setup() -> tuple[Any, Any, Any]:
        g = StructuredGrid.box(size=(20.0, 10.0), shape=(20, 10))
        left = PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0), tol=1e-9)
        tip = Box((20.0, 0.0), (20.0, 0.0), tol=0.6)
        model = LinearElasticity(
            g,
            Material(E=1.0, nu=0.3, rho=1.0),
            supports=[(left, "all")],
            loads=[PointLoad(tip, (0.0, -1.0))],
        )
        return g, model, (DensityFilter(radius=1.5) | SIMP(p=3.0)).bind(g)

    def optimize(opt: object, n_iter: int) -> float:
        g, model, chain = setup()
        opt.setup(n_vars=chain.n_vars, lower=np.zeros(chain.n_vars), upper=np.ones(chain.n_vars))  # type: ignore[attr-defined]
        x = chain.initial_design(0.4)
        comp, con = Compliance(), Volume() <= 0.4
        c = 0.0
        for _ in range(n_iter):
            scale, rho = chain.apply(x), chain.physical_density(x)
            sv = Direct()
            sv.prepare(model.assemble(scale))
            u = np.asarray(sv.solve(model.loads())).reshape(model.n_dof, -1)
            sol = Solution(model=model, mesh=g, displacements=u, interpolated=scale, density=rho)
            c = comp.value(sol)
            dc = chain.pullback(x, comp.grad_field(sol))
            gv = np.array([con.value(sol)])
            dg = chain.pullback_density(x, con.grad_field(sol)).reshape(1, -1)
            res = opt.step(x, c, dc, gv, dg)  # type: ignore[attr-defined]
            x = res.x_next
        return c

    c_oc = optimize(OC(move=0.2), 80)
    c_mma = optimize(MMA(), 80)
    assert abs(c_mma - c_oc) / c_oc < 0.05  # same minimum within 5%


def test_protocol_registry_and_validation() -> None:
    from topokit.optimizers import Optimizer
    from topokit.registry import registry

    opt: Optimizer = MMA()
    assert opt is not None
    assert registry.get("optimizers", "mma") is MMA
    o = _mma(4)
    with pytest.raises(OptimizerError, match="non-finite"):
        o.step(
            np.full(4, 0.5),
            0.0,
            np.array([1.0, np.nan, 1.0, 1.0]),
            np.array([0.0]),
            np.full((1, 4), 0.1),
        )
    with pytest.raises(OptimizerError, match="setup"):
        MMA().step(np.full(4, 0.5), 0.0, -np.ones(4), np.array([0.0]), np.full((1, 4), 0.1))


def test_mma_rejects_degenerate_bounds() -> None:
    opt = MMA()
    with pytest.raises(OptimizerError, match="upper bound must exceed"):
        opt.setup(3, np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 1.0]))


def test_mma_escapes_starting_bound() -> None:
    # MMA's additive update can grow a variable from exactly 0 (OC cannot)
    n = 6
    t = np.ones(n)  # objective min sum (x-1)^2 wants x = 1
    opt = _mma(n)
    x = np.zeros(n)
    for _ in range(60):
        df0 = 2.0 * (x - t)
        res = opt.step(x, float(((x - t) ** 2).sum()), df0, np.array([-1.0]), np.zeros((1, n)))
        x = res.x_next
        if res.change < 1e-7:
            break
    assert x.min() > 0.9  # escaped the lower bound toward the optimum


def test_mma_checkpoint_restore_early_iteration() -> None:
    # restoring from a k=1 state (xold1/xold2 still None) must round-trip
    n = 6
    rng = np.random.default_rng(3)
    t = rng.uniform(0.0, 1.0, n)

    def run(restore_at: int) -> np.ndarray:
        opt = _mma(n)
        x = np.full(n, 0.3)
        for k in range(15):
            if k == restore_at:
                fresh = _mma(n)
                fresh.load_state(opt.state())
                opt = fresh
            df0 = 2.0 * (x - t)
            g = np.array([x.mean() / 0.3 - 1.0])
            dg = np.full((1, n), (1.0 / n) / 0.3)
            x = opt.step(x, 0.0, df0, g, dg).x_next
        return x

    np.testing.assert_array_equal(run(1), run(99))  # restore at k=1 vs never


def test_mma_converges_on_cantilever() -> None:
    from topokit.fem import LinearElasticity, Material, PointLoad
    from topokit.mesh import StructuredGrid
    from topokit.parametrization import SIMP, DensityFilter
    from topokit.responses import Compliance, Solution, Volume
    from topokit.selection import Box, PlaneSlab
    from topokit.solvers import Direct

    g = StructuredGrid.box(size=(20.0, 10.0), shape=(20, 10))
    left = PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0), tol=1e-9)
    tip = Box((20.0, 0.0), (20.0, 0.0), tol=0.6)
    model = LinearElasticity(
        g,
        Material(E=1.0, nu=0.3, rho=1.0),
        supports=[(left, "all")],
        loads=[PointLoad(tip, (0.0, -1.0))],
    )
    chain = (DensityFilter(radius=1.5) | SIMP(p=3.0)).bind(g)
    opt = _mma(chain.n_vars)
    x = chain.initial_design(0.4)
    comp, con = Compliance(), Volume() <= 0.4
    converged = False
    for _ in range(150):
        scale, rho = chain.apply(x), chain.physical_density(x)
        sv = Direct()
        sv.prepare(model.assemble(scale))
        u = np.asarray(sv.solve(model.loads())).reshape(model.n_dof, -1)
        sol = Solution(model=model, mesh=g, displacements=u, interpolated=scale, density=rho)
        dc = chain.pullback(x, comp.grad_field(sol))
        dg = chain.pullback_density(x, con.grad_field(sol)).reshape(1, -1)
        res = opt.step(x, comp.value(sol), dc, np.array([con.value(sol)]), dg)
        x = res.x_next
        if res.change < 1e-3:
            converged = True
            break
    assert converged  # MMA reaches the convergence threshold on the real problem


def test_mma_reports_kkt_decreasing_on_convex() -> None:
    # MMA on a convex 2-var problem; kkt residual should drop toward the optimum
    opt = MMA()
    opt.setup(2, np.zeros(2), np.ones(2))
    x = np.array([0.5, 0.5])
    kkts = []
    for _ in range(20):
        f0 = float(((x - 0.2) ** 2).sum())
        df0 = 2.0 * (x - 0.2)
        g = np.array([x.sum() - 1.0])
        dg = np.ones((1, 2))
        r = opt.step(x, f0, df0, g, dg)
        kkts.append(r.kkt)
        x = r.x_next
    assert np.isfinite(kkts[-1])
    assert kkts[-1] < kkts[0]
