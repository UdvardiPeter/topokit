# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""The unified parametrization chain.

Everything between raw optimizer variables and the field the physics
consumes is a chain of links, each a frozen spec with ``apply`` and
``pullback`` (vector-Jacobian product) once bound to a mesh. Compose with
``|`` and bind::

    chain = SymmetryMap(planes=("x",)) | DensityFilter(radius=1.5) | Heaviside() | SIMP()
    bound = chain.bind(mesh)
    scale = bound.apply(x)
    grad_x = bound.pullback(x, grad_scale)

Design variables live on design elements (reduced further by symmetry).
The bound chain embeds them into the full grid with solid pinned to 1 and
void to 0, runs the density links, re-pins, and applies the terminal
material-interpolation link, which produces the ``FieldSpec`` the physics
model declares. Pinning has zero pullback at pinned positions, so the
chain rule stays exact.

Every link's pullback is finite-difference verified through the registry
meta-test; links whose pullback is deliberately not a VJP (the classic
sensitivity filter) declare ``fd_exempt`` with a reason.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from topokit.backend import get_kernel, register_kernel
from topokit.fields import FieldSpec
from topokit.mesh import StructuredGrid

_F64 = npt.NDArray[np.float64]
_I64 = npt.NDArray[np.int64]

STIFFNESS_SCALE = FieldSpec("stiffness_scale")


class ParametrizationError(ValueError):
    """Invalid chain composition or input."""


def _separable_correlate(grid: _F64, axis_weights: list[_F64]) -> _F64:
    """Correlate with a tensor-product kernel, zero outside the domain."""
    out = grid
    for axis, weights in enumerate(axis_weights):
        half = (len(weights) - 1) // 2
        acc = np.zeros_like(out)
        for k, w in zip(range(-half, half + 1), weights, strict=True):
            if w == 0.0:
                continue
            src = [slice(None)] * out.ndim
            dst = [slice(None)] * out.ndim
            if k < 0:
                src[axis], dst[axis] = slice(-k, None), slice(None, k)
            elif k > 0:
                src[axis], dst[axis] = slice(None, -k), slice(k, None)
            acc[tuple(dst)] += w * out[tuple(src)]
        out = acc
    return out


register_kernel("separable_correlate", "generic", _separable_correlate)


@dataclass(frozen=True)
class LinkSpec:
    """Base for chain links; compose with ``|``."""

    fd_exempt: ClassVar[str | None] = None
    is_terminal: ClassVar[bool] = False
    is_reduced_input: ClassVar[bool] = False

    def __or__(self, other: LinkSpec | Chain) -> Chain:
        if isinstance(other, Chain):
            return Chain((self, *other.links))
        return Chain((self, other))

    def bind(self, mesh: StructuredGrid) -> BoundChain:
        """Bind a single-link chain; equivalent to ``Chain((self,)).bind(mesh)``."""
        return Chain((self,)).bind(mesh)

    def _build(self, mesh: StructuredGrid) -> _BoundLink:
        """Construct the bound link; implemented by each link."""
        raise NotImplementedError

    @classmethod
    def fd_example(cls, mesh: StructuredGrid) -> LinkSpec:
        """Return an instance for the registry-wide FD meta-test."""
        raise NotImplementedError


class _BoundLink:
    def apply(self, x: _F64) -> _F64:
        raise NotImplementedError

    def pullback(self, x: _F64, grad_out: _F64) -> _F64:
        raise NotImplementedError


@dataclass(frozen=True)
class Chain:
    """An ordered tuple of link specs."""

    links: tuple[LinkSpec, ...]

    def __or__(self, other: LinkSpec | Chain) -> Chain:
        if isinstance(other, Chain):
            return Chain((*self.links, *other.links))
        return Chain((*self.links, other))

    def bind(self, mesh: StructuredGrid) -> BoundChain:
        """Validate the composition and bind every link to ``mesh``."""
        links = self.links
        if not links or not links[-1].is_terminal:
            raise ParametrizationError("the chain must end in a terminal link (e.g. SIMP)")
        for link in links[:-1]:
            if link.is_terminal:
                raise ParametrizationError("terminal link must be last and unique")
        for link in links[1:]:
            if link.is_reduced_input:
                raise ParametrizationError(f"{type(link).__name__} must be the first link")
        kinds = {type(link).__name__ for link in links}
        if "DensityFilter" in kinds and "SensitivityFilter" in kinds:
            warnings.warn(
                "combining DensityFilter with SensitivityFilter mixes a real filter "
                "with a gradient heuristic; use one of them",
                stacklevel=2,
            )
        reduced = links[0]._build(mesh) if links[0].is_reduced_input else None
        middle_specs = links[1:-1] if reduced is not None else links[:-1]
        middle = [link._build(mesh) for link in middle_specs]
        terminal = links[-1]._build(mesh)
        return BoundChain(mesh, reduced, middle, terminal, self)


class BoundChain:
    """A chain bound to a mesh; the object the orchestration layer runs."""

    def __init__(
        self,
        mesh: StructuredGrid,
        reduced: _BoundLink | None,
        middle: list[_BoundLink],
        terminal: _BoundLink,
        spec: Chain,
    ) -> None:
        self.mesh = mesh
        self.spec = spec
        self._reduced = reduced
        self._middle = middle
        self._terminal = terminal
        self.out_field: FieldSpec = terminal.out_field  # type: ignore[attr-defined]
        self.n_vars = (
            reduced.n_reduced  # type: ignore[attr-defined]
            if reduced is not None
            else int(mesh.design.sum())
        )
        if self.n_vars == 0:
            raise ParametrizationError(
                "the design region is empty (all elements are solid or void); "
                "at least one design element is required"
            )

    def initial_design(self, volume_fraction: float) -> _F64:
        """Return a uniform starting design."""
        return np.full(self.n_vars, float(volume_fraction))

    def _check(self, x: Any) -> _F64:
        arr = np.asarray(x, dtype=np.float64)
        if arr.shape != (self.n_vars,):
            raise ParametrizationError(f"x shape {arr.shape} != ({self.n_vars},)")
        if not np.isfinite(arr).all():
            raise ParametrizationError("x contains non-finite values")
        # Design variables are densities in [0, 1]; the optimizer enforces this
        # bound, so a violation means a diverged optimizer or direct misuse.
        # A small tolerance absorbs float noise at the bounds.
        if arr.min() < -1e-6 or arr.max() > 1.0 + 1e-6:
            raise ParametrizationError(
                f"design variables must be in [0, 1], got [{arr.min():.4g}, {arr.max():.4g}]"
            )
        return arr

    def _embed(self, design_values: _F64) -> _F64:
        rho = np.zeros(self.mesh.n_elements)
        rho[self.mesh.design] = design_values
        rho[self.mesh.solid] = 1.0
        return rho

    def _pin(self, rho: _F64) -> _F64:
        out = rho.copy()
        out[self.mesh.solid] = 1.0
        out[self.mesh.void] = 0.0
        return out

    def _forward(self, x: _F64) -> tuple[list[_F64], _F64]:
        """Return per-link inputs and the pre-terminal pinned field."""
        inputs: list[_F64] = []
        y = x
        if self._reduced is not None:
            inputs.append(y)
            y = self._reduced.apply(y)
        rho = self._embed(y)
        for link in self._middle:
            inputs.append(rho)
            rho = link.apply(rho)
        pinned = self._pin(rho)
        inputs.append(pinned)
        return inputs, pinned

    def apply(self, x: Any) -> _F64:
        """Map design variables to the physics field, shape ``(n_elements,)``."""
        arr = self._check(x)
        _, pinned = self._forward(arr)
        return self._terminal.apply(pinned)

    def physical_density(self, x: Any) -> _F64:
        """Return the physical density field, ``(n_elements,)``, after filters and projection.

        This is the field before the terminal material-interpolation link
        (solid pinned to 1, void to 0). Responses defined on density, such
        as volume, consume it; responses on the interpolated property, such
        as compliance, consume :meth:`apply`. It is also the branch point
        for coupled physics (one terminal link per physics share it).
        """
        arr = self._check(x)
        _, pinned = self._forward(arr)
        return pinned

    def pullback(self, x: Any, grad_field: Any) -> _F64:
        """Chain-rule ``dF/d(field)`` back to ``dF/dx`` through the whole chain."""
        arr = self._check(x)
        grad = self._check_grad(grad_field)
        inputs, pinned = self._forward(arr)
        g = self._terminal.pullback(pinned, grad)
        return self._density_pullback(inputs, g)

    def pullback_density(self, x: Any, grad_density: Any) -> _F64:
        """Chain-rule ``dF/d(rho_bar)`` back to ``dF/dx`` for density responses.

        Stops at the terminal boundary: use this for a response defined on
        :meth:`physical_density` (e.g. volume), :meth:`pullback` for one on
        :meth:`apply`.
        """
        arr = self._check(x)
        grad = self._check_grad(grad_density)
        inputs, _ = self._forward(arr)
        return self._density_pullback(inputs, grad)

    def _check_grad(self, grad: Any) -> _F64:
        arr = np.asarray(grad, dtype=np.float64)
        if arr.shape != (self.mesh.n_elements,):
            raise ParametrizationError(f"grad shape {arr.shape} != ({self.mesh.n_elements},)")
        return arr

    def _density_pullback(self, inputs: list[_F64], g: _F64) -> _F64:
        """Pull a gradient w.r.t. ``rho_bar`` back to ``x`` (density links only)."""
        g = self._pin_pullback(g)
        idx = len(inputs) - 2  # entry before the appended pinned field
        for link in reversed(self._middle):
            g = link.pullback(inputs[idx], g)
            idx -= 1
        g_design = g[self.mesh.design]
        if self._reduced is not None:
            g_design = self._reduced.pullback(inputs[0], g_design)
        return g_design

    def _pin_pullback(self, g: _F64) -> _F64:
        out = g.copy()
        out[self.mesh.solid] = 0.0
        out[self.mesh.void] = 0.0
        return out


@dataclass(frozen=True)
class SymmetryMap(LinkSpec):
    """Mirror symmetry about domain-center planes; reduces the design space."""

    planes: tuple[str, ...] = ("x",)
    is_reduced_input: ClassVar[bool] = True

    def __post_init__(self) -> None:
        valid = {"x", "y", "z"}
        if not self.planes or any(p not in valid for p in self.planes):
            raise ParametrizationError(f"plane names must be from {sorted(valid)}")

    def _build(self, mesh: StructuredGrid) -> _BoundSymmetry:
        """Build orbit maps and validate mask symmetry."""
        axes = []
        names = "xyz"[: mesh.dim]
        for p in self.planes:
            if p not in names:
                raise ParametrizationError(f"plane {p!r} invalid for a {mesh.dim}D mesh")
            axes.append(names.index(p))
        return _BoundSymmetry(mesh, tuple(sorted(set(axes))))

    @classmethod
    def fd_example(cls, mesh: StructuredGrid) -> SymmetryMap:
        """FD meta-test instance."""
        return cls(planes=("x",))


class _BoundSymmetry(_BoundLink):
    def __init__(self, mesh: StructuredGrid, axes: tuple[int, ...]) -> None:
        ids = np.arange(mesh.n_elements, dtype=np.int64)
        grids = [
            g.ravel(order="F")
            for g in np.meshgrid(*[np.arange(s) for s in mesh.shape], indexing="ij")
        ]
        canonical = ids.copy()
        for subset in range(1, 2 ** len(axes)):
            mapped = [g.copy() for g in grids]
            for bit, axis in enumerate(axes):
                if subset >> bit & 1:
                    mapped[axis] = mesh.shape[axis] - 1 - mapped[axis]
            flat = mapped[0]
            mult = 1
            for a in range(1, mesh.dim):
                mult *= mesh.shape[a - 1]
                flat = flat + mult * mapped[a]
            canonical = np.minimum(canonical, flat)
            for mask in (mesh.design, mesh.solid, mesh.void):
                if not np.array_equal(mask, mask[flat]):
                    raise ParametrizationError(
                        "masks must be symmetric under the requested symmetry planes"
                    )
        design_ids = np.flatnonzero(mesh.design)
        canon_design = canonical[design_ids]
        unique, orbit = np.unique(canon_design, return_inverse=True)
        self.n_reduced = int(unique.size)
        self._orbit: _I64 = orbit.astype(np.int64)

    def apply(self, x: _F64) -> _F64:
        return x[self._orbit]

    def pullback(self, x: _F64, grad_out: _F64) -> _F64:
        out = np.zeros(self.n_reduced)
        np.add.at(out, self._orbit, grad_out)
        return out


@dataclass(frozen=True)
class DensityFilter(LinkSpec):
    """Separable triangular-kernel density filter with mask-aware normalization.

    The kernel is the tensor product of per-axis triangles of physical
    ``radius`` (the spec's deliberate separable approximation of the radial
    cone, chosen for its O(n*r) cost). Void is excluded; solid participates
    pinned at 1. The pullback is the exact transpose of the normalized
    correlation.

    Because the kernel is separable, its reach is ``sqrt(dim)`` larger along
    diagonals than along axes (measured 1.41x in 2D, 1.73x in 3D), so the
    minimum member size it enforces is orientation-dependent, tightest for
    axis-aligned features. A radial kernel would be needed for isotropic
    control; the radius-to-member-size calibration is set in the
    manufacturing-constraint work.
    """

    radius: float = 1.5

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ParametrizationError(f"radius must be > 0, got {self.radius}")

    def _build(self, mesh: StructuredGrid) -> _BoundDensityFilter:
        """Precompute kernels and the mask normalization."""
        return _BoundDensityFilter(mesh, self.radius)

    @classmethod
    def fd_example(cls, mesh: StructuredGrid) -> DensityFilter:
        """FD meta-test instance."""
        return cls(radius=1.6 * max(mesh.spacing))


class _BoundDensityFilter(_BoundLink):
    def __init__(self, mesh: StructuredGrid, radius: float) -> None:
        self.mesh = mesh
        self._weights: list[_F64] = []
        for axis in range(mesh.dim):
            h = mesh.spacing[axis]
            half = max(0, math.ceil(radius / h) - 1)
            offs = np.arange(-half, half + 1)
            w = np.maximum(0.0, 1.0 - np.abs(offs) * h / radius)
            self._weights.append(w)
        self._correlate = get_kernel("separable_correlate", "numpy")
        active = mesh.active_elements.astype(np.float64)
        self._active = active
        denom = self._correlate(mesh.to_grid(active), self._weights)
        self._denom = np.maximum(mesh.to_flat(denom), 1e-300)

    def apply(self, x: _F64) -> _F64:
        mesh = self.mesh
        num = self._correlate(mesh.to_grid(x * self._active), self._weights)
        out = mesh.to_flat(num) / self._denom
        return np.where(mesh.active_elements, out, 0.0)

    def pullback(self, x: _F64, grad_out: _F64) -> _F64:
        # Forward is F = A D^-1 C A (mask, normalize, correlate, mask), with
        # A and D diagonal and the separable correlation C symmetric. The
        # exact transpose is A C D^-1 A; the diagonal factors commute, so the
        # code below (A C A D^-1) equals it. Keep that commutativity in mind
        # before reordering these operations.
        mesh = self.mesh
        scaled = np.where(mesh.active_elements, grad_out / self._denom, 0.0)
        back = self._correlate(mesh.to_grid(scaled), self._weights)
        return mesh.to_flat(back) * self._active


@dataclass(frozen=True)
class Heaviside(LinkSpec):
    """Smooth tanh projection sharpening intermediate densities."""

    beta: float = 1.0
    eta: float = 0.5

    def __post_init__(self) -> None:
        if self.beta <= 0.0:
            raise ParametrizationError(f"beta must be > 0, got {self.beta}")
        if not 0.0 < self.eta < 1.0:
            raise ParametrizationError(f"eta must be in (0, 1), got {self.eta}")

    def _build(self, mesh: StructuredGrid) -> _BoundHeaviside:
        """Bind; the projection is elementwise and mesh-independent."""
        return _BoundHeaviside(self.beta, self.eta)

    @classmethod
    def fd_example(cls, mesh: StructuredGrid) -> Heaviside:
        """FD meta-test instance."""
        return cls(beta=3.0)


class _BoundHeaviside(_BoundLink):
    def __init__(self, beta: float, eta: float) -> None:
        self.beta = beta
        self.eta = eta
        self._den = math.tanh(beta * eta) + math.tanh(beta * (1.0 - eta))

    def apply(self, x: _F64) -> _F64:
        num = np.tanh(self.beta * self.eta) + np.tanh(self.beta * (x - self.eta))
        return np.asarray(num / self._den, dtype=np.float64)

    def pullback(self, x: _F64, grad_out: _F64) -> _F64:
        d = self.beta * (1.0 - np.tanh(self.beta * (x - self.eta)) ** 2) / self._den
        return np.asarray(grad_out * d, dtype=np.float64)


@dataclass(frozen=True)
class SIMP(LinkSpec):
    """Modified SIMP material interpolation; the terminal link for elasticity."""

    p: float = 3.0
    scale_min: float = 1e-9
    is_terminal: ClassVar[bool] = True

    def __post_init__(self) -> None:
        # p >= 1 is SIMP's definition (the penalization exponent); p < 1 also
        # gives an unbounded gradient at rho = 0, which poisons the optimizer.
        if self.p < 1.0:
            raise ParametrizationError(f"p must be >= 1, got {self.p}")
        if not 0.0 <= self.scale_min < 1.0:
            raise ParametrizationError(f"scale_min must be in [0, 1), got {self.scale_min}")

    def _build(self, mesh: StructuredGrid) -> _BoundSIMP:
        """Bind; the interpolation is elementwise."""
        return _BoundSIMP(self.p, self.scale_min)

    @classmethod
    def fd_example(cls, mesh: StructuredGrid) -> SIMP:
        """FD meta-test instance."""
        return cls(p=3.0)


class _BoundSIMP(_BoundLink):
    out_field = STIFFNESS_SCALE

    def __init__(self, p: float, scale_min: float) -> None:
        self.p = p
        self.scale_min = scale_min

    def apply(self, x: _F64) -> _F64:
        return np.asarray(self.scale_min + x**self.p * (1.0 - self.scale_min))

    def pullback(self, x: _F64, grad_out: _F64) -> _F64:
        return np.asarray(grad_out * self.p * x ** (self.p - 1.0) * (1.0 - self.scale_min))


@dataclass(frozen=True)
class SensitivityFilter(LinkSpec):
    """Sigmund's classic sensitivity filter.

    Identity in the forward direction; the pullback smooths gradients with
    density weighting. It is a gradient heuristic, deliberately not the
    VJP of any map, hence ``fd_exempt``. Density filtering plus projection
    is the recommended modern alternative.
    """

    radius: float = 1.5
    fd_exempt: ClassVar[str | None] = (
        "gradient heuristic by definition; its pullback is not the VJP of apply"
    )

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ParametrizationError(f"radius must be > 0, got {self.radius}")

    def _build(self, mesh: StructuredGrid) -> _BoundSensitivityFilter:
        """Bind, reusing the density-filter kernels."""
        return _BoundSensitivityFilter(mesh, self.radius)

    @classmethod
    def fd_example(cls, mesh: StructuredGrid) -> SensitivityFilter:
        """Unused; the link is fd_exempt."""
        return cls()


class _BoundSensitivityFilter(_BoundLink):
    def __init__(self, mesh: StructuredGrid, radius: float) -> None:
        self._inner = _BoundDensityFilter(mesh, radius)
        self.mesh = mesh

    def apply(self, x: _F64) -> _F64:
        return x

    def pullback(self, x: _F64, grad_out: _F64) -> _F64:
        mesh = self.mesh
        inner = self._inner
        num = inner._correlate(mesh.to_grid(x * grad_out * inner._active), inner._weights)
        weighted = mesh.to_flat(num) / (inner._denom * np.maximum(x, 1e-3))
        return np.where(mesh.active_elements, weighted, 0.0)
