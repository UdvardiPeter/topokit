"""Tests for the optimality-criteria optimizer."""

import numpy as np
import pytest

from topokit.optimizers import OC, Optimizer, OptimizerError
from topokit.optimizers._base import StepResult, kkt_residual


def _oc(n: int, move: float = 0.2, eta: float = 0.5) -> OC:
    opt = OC(move=move, eta=eta)
    opt.setup(n_vars=n, lower=np.zeros(n), upper=np.ones(n))
    return opt


def test_protocol_and_registry() -> None:
    from topokit.registry import registry

    opt: Optimizer = OC()
    assert opt is not None
    assert registry.get("optimizers", "oc") is OC


def test_step_satisfies_volume_constraint() -> None:
    # 6 vars, uniform compliance sensitivity, volume constraint sum(x)/6 <= 0.5
    n = 6
    opt = _oc(n)
    x = np.full(n, 0.5)
    df0 = -np.ones(n)  # dc/dx < 0 (more material -> less compliance)
    g = np.array([x.mean() / 0.5 - 1.0])  # volume <= 0.5, normalized
    dg = np.full((1, n), (1.0 / n) / 0.5)
    res = opt.step(x, float(x.mean()), df0, g, dg)
    # uniform sensitivities -> constraint active at the target volume
    assert res.x_next.mean() == pytest.approx(0.5, abs=1e-3)


def test_higher_sensitivity_gets_more_material() -> None:
    n = 4
    opt = _oc(n)
    x = np.full(n, 0.5)
    df0 = -np.array([4.0, 1.0, 1.0, 1.0])  # element 0 most sensitive
    dg = np.full((1, n), (1.0 / n) / 0.5)
    g = np.array([x.mean() / 0.5 - 1.0])
    res = opt.step(x, 0.0, df0, g, dg)
    assert res.x_next[0] > res.x_next[1]
    assert res.x_next.mean() == pytest.approx(0.5, abs=1e-3)


def test_move_limit_respected() -> None:
    n = 4
    opt = _oc(n, move=0.1)
    x = np.full(n, 0.5)
    df0 = -np.array([100.0, 1e-6, 1e-6, 1e-6])  # element 0 wants to jump up hard
    dg = np.full((1, n), (1.0 / n) / 0.5)
    g = np.array([x.mean() / 0.5 - 1.0])
    res = opt.step(x, 0.0, df0, g, dg)
    assert res.x_next[0] <= 0.5 + 0.1 + 1e-9


def test_box_bounds_respected() -> None:
    n = 4
    opt = _oc(n, move=1.0)
    x = np.full(n, 0.9)
    df0 = -np.array([100.0, 1.0, 1.0, 1.0])
    dg = np.full((1, n), (1.0 / n) / 0.9)
    g = np.array([x.mean() / 0.9 - 1.0])
    res = opt.step(x, 0.0, df0, g, dg)
    assert res.x_next.max() <= 1.0 + 1e-12
    assert res.x_next.min() >= 0.0 - 1e-12


def test_change_is_max_abs_step() -> None:
    n = 4
    opt = _oc(n, move=0.2)
    x = np.full(n, 0.5)
    df0 = -np.array([10.0, 1e-9, 1e-9, 1e-9])
    dg = np.full((1, n), (1.0 / n) / 0.5)
    g = np.array([x.mean() / 0.5 - 1.0])
    res = opt.step(x, 0.0, df0, g, dg)
    assert res.change == pytest.approx(float(np.abs(res.x_next - x).max()))


def test_requires_single_constraint() -> None:
    opt = _oc(4)
    x = np.full(4, 0.5)
    df0 = -np.ones(4)
    g = np.array([0.0, 0.0])
    dg = np.ones((2, 4))
    with pytest.raises(OptimizerError, match="one constraint"):
        opt.step(x, 0.0, df0, g, dg)


def test_negative_constraint_gradient_raises() -> None:
    opt = _oc(4)
    x = np.full(4, 0.5)
    df0 = -np.ones(4)
    g = np.array([0.0])
    dg = -np.ones((1, 4))  # constraint decreasing in density: OC can't bisect
    with pytest.raises(OptimizerError, match="non-negative"):
        opt.step(x, 0.0, df0, g, dg)


def test_state_roundtrip() -> None:
    opt = _oc(4)
    assert opt.state() == {}
    opt.load_state({})  # no-op for stateless OC


def test_converges_on_separable_quadratic() -> None:
    # minimize sum c_e * (1 - x_e)  [linear, want x high] s.t. mean(x) <= vf
    # OC fixed point: highest-c elements saturate to 1 until the budget is spent.
    n = 8
    vf = 0.4
    c = np.array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    opt = _oc(n, move=0.2)
    x = np.full(n, vf)
    for _ in range(200):
        df0 = -c  # d/dx of -c*x  (objective sum -c_e x_e, minimize)
        g = np.array([x.mean() / vf - 1.0])
        dg = np.full((1, n), (1.0 / n) / vf)
        res = opt.step(x, 0.0, df0, g, dg)
        if res.change < 1e-6:
            x = res.x_next
            break
        x = res.x_next
    assert x.mean() == pytest.approx(vf, abs=1e-3)
    # the cheapest elements (low c) should be emptied, expensive ones filled
    assert x[0] > x[-1]
    assert np.argmax(x) == 0


def test_oc_minimizes_cantilever_end_to_end() -> None:
    # First real optimization: full loop chain -> fem -> solver -> responses -> OC.
    # Filter + SIMP (no projection) keeps volume linear in x, so OC tracks the
    # target exactly each step, matching the classic 88-line behavior.
    from topokit.fem import LinearElasticity, Material, PointLoad
    from topokit.mesh import StructuredGrid
    from topokit.parametrization import SIMP, DensityFilter
    from topokit.responses import Compliance, Solution, Volume
    from topokit.selection import Box, PlaneSlab
    from topokit.solvers import Direct

    g = StructuredGrid.box(size=(20.0, 10.0), shape=(20, 10))
    left = PlaneSlab(point=(0.0, 0.0), normal=(1.0, 0.0), tol=1e-9)
    tip = Box((20.0, 0.0), (20.0, 0.0), tol=0.6)  # bottom-right corner load
    model = LinearElasticity(
        g,
        Material(E=1.0, nu=0.3, rho=1.0),
        supports=[(left, "all")],
        loads=[PointLoad(tip, (0.0, -1.0))],
    )
    chain = (DensityFilter(radius=1.5) | SIMP(p=3.0)).bind(g)
    vf = 0.4
    opt = OC(move=0.2)
    opt.setup(n_vars=chain.n_vars, lower=np.zeros(chain.n_vars), upper=np.ones(chain.n_vars))
    x = chain.initial_design(vf)
    comp, con = Compliance(), Volume() <= vf

    c_hist, vol_hist = [], []
    for _ in range(80):
        scale, rho = chain.apply(x), chain.physical_density(x)
        solver = Direct()
        solver.prepare(model.assemble(scale))
        u = np.asarray(solver.solve(model.loads())).reshape(model.n_dof, -1)
        sol = Solution(model=model, mesh=g, displacements=u, interpolated=scale, density=rho)
        c_hist.append(comp.value(sol))
        vol_hist.append(Volume().value(sol))
        dc = chain.pullback(x, comp.grad_field(sol))
        gv = np.array([con.value(sol)])
        dg = chain.pullback_density(x, con.grad_field(sol)).reshape(1, -1)
        res = opt.step(x, c_hist[-1], dc, gv, dg)
        x = res.x_next
        if res.change < 1e-3:
            break

    assert c_hist[-1] < 0.5 * c_hist[0]  # compliance substantially reduced
    # volume is linear in x here (no projection), so OC's bisection tracks the
    # target near-exactly every iteration (a tight guard against drift bugs)
    assert vol_hist[-1] == pytest.approx(vf, abs=1e-3)
    assert max(abs(v - vf) for v in vol_hist[1:]) < 1e-3
    # material redistributed to both extremes (a structure formed); crisp 0/1
    # boundaries need Heaviside projection, which this no-projection chain omits
    rho_d = chain.physical_density(x)[g.design]
    assert rho_d.min() < 0.1
    assert rho_d.max() > 0.9


def test_eta_not_half_branch() -> None:
    # the b**eta path (eta != 0.5) is distinct code from the sqrt fast path
    n = 4
    x = np.full(n, 0.5)
    df0 = -np.array([4.0, 1.0, 1.0, 1.0])
    dg = np.full((1, n), (1.0 / n) / 0.5)
    g = np.array([0.0])
    gentle = _oc(n, eta=0.3, move=1.0).step(x, 0.0, df0, g, dg)
    aggressive = _oc(n, eta=1.0, move=1.0).step(x, 0.0, df0, g, dg)
    assert np.isfinite(gentle.x_next).all() and np.isfinite(aggressive.x_next).all()
    assert gentle.x_next.mean() == pytest.approx(0.5, abs=1e-3)
    assert aggressive.x_next.mean() == pytest.approx(0.5, abs=1e-3)
    # larger eta pushes the most-sensitive element harder
    assert aggressive.x_next[0] > gentle.x_next[0]


def test_input_validation() -> None:
    opt = _oc(4)
    x = np.full(4, 0.5)
    dg = np.full((1, 4), 0.25)
    g = np.array([0.0])
    with pytest.raises(OptimizerError, match="x shape"):
        opt.step(np.full(6, 0.5), 0.0, -np.ones(6), g, np.full((1, 6), 0.25))
    with pytest.raises(OptimizerError, match="df0 shape"):
        opt.step(x, 0.0, -np.ones(3), g, dg)
    with pytest.raises(OptimizerError, match="non-finite"):
        opt.step(x, 0.0, -np.array([1.0, np.nan, 1.0, 1.0]), g, dg)
    with pytest.raises(OptimizerError, match="non-finite"):
        opt.step(x, 0.0, -np.ones(4), g, np.full((1, 4), np.inf))


def test_step_before_setup_raises() -> None:
    opt = OC()
    with pytest.raises(OptimizerError, match="setup"):
        opt.step(np.full(4, 0.5), 0.0, -np.ones(4), np.array([0.0]), np.full((1, 4), 0.25))


def test_kkt_residual_zero_at_unconstrained_interior_optimum() -> None:
    # interior point, zero gradient, no constraints -> residual 0
    x = np.array([0.5, 0.5])
    res = kkt_residual(
        x,
        df0=np.zeros(2),
        g=np.zeros(0),
        dg=np.zeros((0, 2)),
        lower=np.zeros(2),
        upper=np.ones(2),
        lam=np.zeros(0),
    )
    assert res == 0.0


def test_kkt_residual_flags_feasibility_and_stationarity() -> None:
    x = np.array([0.5])
    # violated constraint g=0.3 (>0) dominates
    res = kkt_residual(
        x,
        df0=np.zeros(1),
        g=np.array([0.3]),
        dg=np.zeros((1, 1)),
        lower=np.zeros(1),
        upper=np.ones(1),
        lam=np.zeros(1),
    )
    assert res == 0.3
    # nonzero Lagrangian gradient at an interior point -> stationarity residual
    res2 = kkt_residual(
        x,
        df0=np.array([0.2]),
        g=np.zeros(1),
        dg=np.zeros((1, 1)),
        lower=np.zeros(1),
        upper=np.ones(1),
        lam=np.zeros(1),
    )
    assert res2 == pytest.approx(0.2)


def test_step_result_kkt_defaults_zero() -> None:
    assert StepResult(x_next=np.zeros(1), change=0.0).kkt == 0.0


def test_oc_reports_kkt_decreasing() -> None:
    # a small convex compliance-like problem; kkt should fall as OC converges
    n = 8
    opt = _oc(n)
    x = np.full(n, 0.5)
    df0 = -np.arange(1.0, n + 1.0)  # increasing variables lowers f
    dg = np.ones((1, n)) / n  # volume
    first = last = None
    for _ in range(40):
        g = np.array([x.mean() - 0.4])
        r = opt.step(x, float(-(df0 * x).sum()), df0, g, dg)
        first = r.kkt if first is None else first
        last = r.kkt
        x = r.x_next
    assert last is not None and first is not None
    assert np.isfinite(last)
    assert last < first
