# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Clean-room Method of Moving Asymptotes (Svanberg 2002, standard non-GC MMA).

Implemented from the published equations (see the WP-1.8b plan), not from
any reference code. ``solve_subproblem`` is a primal-dual interior-point
solver for the convex separable MMA subproblem; ``MMA`` wraps it with the
moving-asymptote update and the standard topology-optimization artificial-
variable parameters (a0=1, a=0, c=1000, d=1).

Correctness is established by the test suite: the subproblem solver is
checked against its KKT conditions and cross-checked against scipy SLSQP,
and the full MMA is matched to scipy on convex problems and to OC on the
cantilever.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from topokit.optimizers._base import OptimizerError, StepResult, check_step_inputs

_F64 = npt.NDArray[np.float64]


@dataclass(frozen=True)
class MMASubproblem:
    """A convex separable MMA subproblem (the approximation at one iterate)."""

    low: _F64
    upp: _F64
    alpha: _F64
    beta: _F64
    p0: _F64
    q0: _F64
    p: _F64  # (m, n)
    q: _F64  # (m, n)
    b: _F64  # (m,)
    a0: float
    a: _F64  # (m,)
    c: _F64  # (m,)
    d: _F64  # (m,)

    @property
    def n(self) -> int:
        """Number of design variables."""
        return int(self.p0.size)

    @property
    def m(self) -> int:
        """Number of constraints."""
        return int(self.b.size)


@dataclass(frozen=True)
class SubproblemSolution:
    """The primal solution of an MMA subproblem."""

    x: _F64
    y: _F64
    z: float
    lam: _F64


def solve_subproblem(sp: MMASubproblem) -> SubproblemSolution:
    """Solve the MMA subproblem by a primal-dual interior-point method."""
    n, m = sp.n, sp.m
    alpha, beta, low, upp = sp.alpha, sp.beta, sp.low, sp.upp
    eem = np.ones(m)

    x = 0.5 * (alpha + beta)
    y = eem.copy()
    z = 1.0
    lam = eem.copy()
    xsi = np.maximum(1.0, 1.0 / (x - alpha))
    eta = np.maximum(1.0, 1.0 / (beta - x))
    mu = np.maximum(eem, 0.5 * sp.c)
    zet = 1.0
    s = eem.copy()

    eps = 1.0
    while eps > 1e-7:
        for _ in range(200):
            ux = upp - x
            xl = x - low
            plam = sp.p0 + lam @ sp.p
            qlam = sp.q0 + lam @ sp.q
            gvec = sp.p @ (1.0 / ux) + sp.q @ (1.0 / xl)
            dpsidx = plam / ux**2 - qlam / xl**2

            rex = dpsidx - xsi + eta
            rey = sp.c + sp.d * y - lam - mu
            rez = sp.a0 - zet - sp.a @ lam
            relam = gvec - sp.a * z - y + s - sp.b
            rexsi = xsi * (x - alpha) - eps
            reeta = eta * (beta - x) - eps
            remu = mu * y - eps
            rezet = zet * z - eps
            res = lam * s - eps
            residu = np.concatenate([rex, rey, [rez], relam, rexsi, reeta, remu, [rezet], res])
            if np.linalg.norm(residu, np.inf) < 0.9 * eps:
                break

            gg = sp.p / ux**2 - sp.q / xl**2  # (m, n)
            delx = dpsidx - eps / (x - alpha) + eps / (beta - x)
            dely = sp.c + sp.d * y - lam - eps / y
            delz = sp.a0 - sp.a @ lam - eps / z
            dellam = gvec - sp.a * z - y - sp.b + eps / lam
            diagx = 2.0 * plam / ux**3 + 2.0 * qlam / xl**3 + xsi / (x - alpha) + eta / (beta - x)
            diagy = sp.d + mu / y
            diaglamyi = s / lam + 1.0 / diagy

            if m > 0:
                ggdx = gg / diagx
                alam = ggdx @ gg.T + np.diag(diaglamyi)
                blam = dellam + dely / diagy - gg @ (delx / diagx)
                aa = np.zeros((m + 1, m + 1))
                aa[:m, :m] = alam
                aa[:m, m] = sp.a
                aa[m, :m] = sp.a
                aa[m, m] = -zet / z
                bb = np.concatenate([blam, [delz]])
                sol = np.linalg.solve(aa, bb)
                dlam = sol[:m]
                dz = float(sol[m])
                dx = -(delx + dlam @ gg) / diagx
                dy = (dlam - dely) / diagy
            else:
                dlam = np.zeros(0)
                dz = float(-delz * z / zet)
                dx = -delx / diagx
                dy = np.zeros(0)

            dxsi = -xsi + (eps - xsi * dx) / (x - alpha)
            deta = -eta + (eps + eta * dx) / (beta - x)
            dmu = -mu + (eps - mu * dy) / y if m > 0 else np.zeros(0)
            dzet = -zet + (eps - zet * dz) / z
            ds = -s + (eps - s * dlam) / lam if m > 0 else np.zeros(0)

            # fraction-to-boundary step
            steps = [1.0]
            steps.append(float((-1.01 * dx / (x - alpha)).max(initial=0.0)))
            steps.append(float((1.01 * dx / (beta - x)).max(initial=0.0)))
            for var, dvar in ((y, dy), (lam, dlam), (mu, dmu), (s, ds)):
                if var.size:
                    steps.append(float((-1.01 * dvar / var).max(initial=0.0)))
            steps.append(-1.01 * dz / z)
            steps.append(-1.01 * dzet / zet)
            if n:
                steps.append(float((-1.01 * deta / eta).max(initial=0.0)))
                steps.append(float((-1.01 * dxsi / xsi).max(initial=0.0)))
            t = 1.0 / max(steps)

            old_norm = np.linalg.norm(residu, np.inf)
            for _ in range(50):
                xn = x + t * dx
                yn = y + t * dy
                zn = z + t * dz
                lamn = lam + t * dlam
                xsin = xsi + t * dxsi
                etan = eta + t * deta
                mun = mu + t * dmu
                zetn = zet + t * dzet
                sn = s + t * ds
                uxn = upp - xn
                xln = xn - low
                plamn = sp.p0 + lamn @ sp.p
                qlamn = sp.q0 + lamn @ sp.q
                gvecn = sp.p @ (1.0 / uxn) + sp.q @ (1.0 / xln)
                dpsidxn = plamn / uxn**2 - qlamn / xln**2
                rn = np.concatenate(
                    [
                        dpsidxn - xsin + etan,
                        sp.c + sp.d * yn - lamn - mun,
                        [sp.a0 - zetn - sp.a @ lamn],
                        gvecn - sp.a * zn - yn + sn - sp.b,
                        xsin * (xn - alpha) - eps,
                        etan * (beta - xn) - eps,
                        mun * yn - eps,
                        [zetn * zn - eps],
                        lamn * sn - eps,
                    ]
                )
                if np.linalg.norm(rn, np.inf) <= old_norm:
                    break
                t *= 0.5
            x, y, z, lam, xsi, eta, mu, zet, s = xn, yn, zn, lamn, xsin, etan, mun, zetn, sn
        eps *= 0.1

    return SubproblemSolution(x=x, y=y, z=float(z), lam=lam)


@dataclass
class MMA:
    """Standard MMA (non-globally-convergent), Svanberg 2002.

    Handles ``m >= 1`` constraints. Stateful: the asymptotes and the two
    previous design points are kept across steps and serialized by
    :meth:`state` for checkpointing.
    """

    asyinit: float = 0.5
    asydecr: float = 0.7
    asyincr: float = 1.2
    asybound: float = 10.0
    move: float = 0.5
    raa0: float = 1e-5

    def __post_init__(self) -> None:
        self._lower: _F64 | None = None
        self._upper: _F64 | None = None
        self._k = 0
        self._xold1: _F64 | None = None
        self._xold2: _F64 | None = None
        self._low: _F64 | None = None
        self._upp: _F64 | None = None

    def setup(self, n_vars: int, lower: _F64, upper: _F64) -> None:
        """Store the box bounds and reset the asymptote history."""
        self._lower = np.asarray(lower, dtype=np.float64)
        self._upper = np.asarray(upper, dtype=np.float64)
        if self._lower.shape != (n_vars,) or self._upper.shape != (n_vars,):
            raise OptimizerError("lower/upper must have shape (n_vars,)")
        self._k = 0
        self._xold1 = None
        self._xold2 = None
        self._low = None
        self._upp = None

    def step(self, x: _F64, f0: float, df0: _F64, g: _F64, dg: _F64) -> StepResult:
        """One MMA step: update asymptotes, build and solve the subproblem."""
        x, df0, g, dg = check_step_inputs(self._lower, x, df0, g, dg)
        assert self._lower is not None and self._upper is not None
        xmin, xmax = self._lower, self._upper
        rng = xmax - xmin
        m = g.size
        self._k += 1

        if self._k <= 2 or self._low is None or self._upp is None:
            low = x - self.asyinit * rng
            upp = x + self.asyinit * rng
        else:
            assert self._xold1 is not None and self._xold2 is not None
            sign = (x - self._xold1) * (self._xold1 - self._xold2)
            gamma = np.ones_like(x)
            gamma[sign < 0.0] = self.asydecr
            gamma[sign > 0.0] = self.asyincr
            low = x - gamma * (self._xold1 - self._low)
            upp = x + gamma * (self._upp - self._xold1)
            low = np.clip(low, x - self.asybound * rng, x - 0.01 * rng)
            upp = np.clip(upp, x + 0.01 * rng, x + self.asybound * rng)

        alpha = np.maximum.reduce([xmin, low + 0.1 * (x - low), x - self.move * rng])
        beta = np.minimum.reduce([xmax, upp - 0.1 * (upp - x), x + self.move * rng])

        ux = upp - x
        xl = x - low
        df0p = np.maximum(df0, 0.0)
        df0n = np.maximum(-df0, 0.0)
        p0 = ux**2 * (1.001 * df0p + 0.001 * df0n + self.raa0 / rng)
        q0 = xl**2 * (0.001 * df0p + 1.001 * df0n + self.raa0 / rng)
        dgp = np.maximum(dg, 0.0)
        dgn = np.maximum(-dg, 0.0)
        p = ux[None, :] ** 2 * (1.001 * dgp + 0.001 * dgn + self.raa0 / rng)
        q = xl[None, :] ** 2 * (0.001 * dgp + 1.001 * dgn + self.raa0 / rng)
        b = (p / ux + q / xl).sum(axis=1) - g

        sp = MMASubproblem(
            low=low,
            upp=upp,
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
        sol = solve_subproblem(sp)

        self._xold2 = self._xold1
        self._xold1 = x.copy()
        self._low = low
        self._upp = upp
        x_next = sol.x
        return StepResult(x_next=x_next, change=float(np.abs(x_next - x).max()))

    def state(self) -> dict[str, Any]:
        """Asymptotes and design history, for checkpointing."""
        return {
            "k": self._k,
            "xold1": None if self._xold1 is None else self._xold1.copy(),
            "xold2": None if self._xold2 is None else self._xold2.copy(),
            "low": None if self._low is None else self._low.copy(),
            "upp": None if self._upp is None else self._upp.copy(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore the asymptote history from :meth:`state`."""
        self._k = int(state["k"])
        self._xold1 = state["xold1"]
        self._xold2 = state["xold2"]
        self._low = state["low"]
        self._upp = state["upp"]
