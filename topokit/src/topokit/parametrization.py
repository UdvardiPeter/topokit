# SPDX-License-Identifier: MIT
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

To add a new link, subclass ``LinkSpec`` and implement ``build``, returning
a ``BoundLink``. The bound link implements ``apply`` and ``pullback``, and
a terminal link's bound form must set ``out_field`` to the field the
physics consumes. Mark a terminal link class with ``is_terminal = True``.

Design variables live on design elements.
The bound chain embeds them into the full mesh with solid pinned to 1 and
void to 0, runs the density links, re-pins, and applies the terminal
material-interpolation link, which produces the ``FieldSpec`` the physics
model declares. Pinning has zero pullback at pinned positions, so the
chain rule stays exact.

Every link's pullback is finite-difference verified through the registry
meta-test.
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from topokit.backend import get_kernel, register_kernel
from topokit.exceptions import TopoKitError
from topokit.fields import FieldSpec
from topokit.mesh import Mesh, StructuredGrid

_F64 = npt.NDArray[np.float64]
_I64 = npt.NDArray[np.int64]

# Warnings raised here are about the caller's configuration, and binding
# happens both directly and from Problem(); skipping the package's own frames
# points every one of them at the user's line instead of an internal bind.
_PACKAGE_DIR = os.path.dirname(__file__)

STIFFNESS_SCALE = FieldSpec("stiffness_scale")


class ParametrizationError(TopoKitError, ValueError):
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


def _radial_correlate(grid: _F64, kernel: _F64) -> _F64:
    """Correlate with a dense N-D kernel, zero outside the domain.

    The kernel is symmetric under reflection (a function of distance), so the
    correlation is self-adjoint and serves as its own pullback.
    """
    out = np.zeros_like(grid)
    halves = [(s - 1) // 2 for s in kernel.shape]
    for idx in np.ndindex(kernel.shape):
        w = float(kernel[idx])
        if w == 0.0:
            continue
        src = [slice(None)] * grid.ndim
        dst = [slice(None)] * grid.ndim
        for axis, (i, half) in enumerate(zip(idx, halves, strict=True)):
            k = i - half
            if k < 0:
                src[axis], dst[axis] = slice(-k, None), slice(None, k)
            elif k > 0:
                src[axis], dst[axis] = slice(None, -k), slice(k, None)
        out[tuple(dst)] += w * grid[tuple(src)]
    return out


register_kernel("radial_correlate", "generic", _radial_correlate)


def _axis_halves(mesh: StructuredGrid, radius: float) -> list[int]:
    """Kernel half-width in elements per axis for a physical ``radius``."""
    return [max(0, math.ceil(radius / mesh.spacing[axis]) - 1) for axis in range(mesh.dim)]


def _unfiltered_axes(mesh: StructuredGrid, halves: list[int]) -> str:
    """Name the axes whose kernel reaches no neighbor, with their element size."""
    return ", ".join(
        f"{'xyz'[axis]} (element size {mesh.spacing[axis]:g})"
        for axis, half in enumerate(halves)
        if half == 0
    )


def _warn_if_unfiltered(radius: float, where: str) -> None:
    """Warn that the bound kernel reaches no neighbor in the directions ``where``.

    A radius at or below the element size leaves a width-1 kernel, i.e. the
    identity: no minimum member size is enforced and checkerboarding is not
    suppressed there. The default radius reaches a neighbor on the unit-spacing
    grids the docs use, so this fires only on a radius that cannot do its job --
    the usual cause being one sized for different length units.
    """
    if not where:
        return
    warnings.warn(
        f"filter radius {radius:g} reaches no neighboring element along {where}: "
        "the filter is the identity there, so it enforces no minimum member size "
        "and suppresses no checkerboarding. Use a radius of at least ~1.5 element "
        "sizes -- radius is a physical length in mesh units, not an element count.",
        skip_file_prefixes=(_PACKAGE_DIR,),
    )


@dataclass(frozen=True)
class LinkSpec:
    """Base for chain links; compose with ``|``."""

    is_terminal: ClassVar[bool] = False

    def __or__(self, other: LinkSpec | Chain) -> Chain:
        if isinstance(other, Chain):
            return Chain((self, *other.links))
        return Chain((self, other))

    def bind(self, mesh: Mesh) -> BoundChain:
        """Bind a single-link chain; equivalent to ``Chain((self,)).bind(mesh)``."""
        return Chain((self,)).bind(mesh)

    def build(self, mesh: Mesh) -> BoundLink:
        """Construct the bound link for ``mesh``; implemented by each link."""
        raise NotImplementedError

    @classmethod
    def fd_example(cls, mesh: Mesh) -> LinkSpec:
        """Return an instance for the registry-wide FD meta-test."""
        raise NotImplementedError


class BoundLink:
    """A link bound to a mesh: the pair of maps a chain composes.

    Implement ``apply`` (forward) and ``pullback`` (the vector-Jacobian
    product; finite-difference verified for registry-registered links).
    A terminal link's bound form must set ``out_field`` to the
    :class:`~topokit.fields.FieldSpec` the physics consumes. A
    reduced-input link's bound form must set ``n_reduced``, the
    size of its input space.
    """

    out_field: FieldSpec | None = None

    def apply(self, x: _F64) -> _F64:
        """Map the link input to its output."""
        raise NotImplementedError

    def pullback(self, x: _F64, grad_out: _F64) -> _F64:
        """Chain-rule a gradient at the output back to the input."""
        raise NotImplementedError


@dataclass(frozen=True)
class Chain:
    """An ordered tuple of link specs."""

    links: tuple[LinkSpec, ...]

    def __or__(self, other: LinkSpec | Chain) -> Chain:
        if isinstance(other, Chain):
            return Chain((*self.links, *other.links))
        return Chain((*self.links, other))

    def bind(self, mesh: Mesh) -> BoundChain:
        """Validate the composition and bind every link to ``mesh``."""
        links = self.links
        if not links or not links[-1].is_terminal:
            raise ParametrizationError("the chain must end in a terminal link (e.g. SIMP)")
        for link in links[:-1]:
            if link.is_terminal:
                raise ParametrizationError("terminal link must be last and unique")
        middle = [link.build(mesh) for link in links[:-1]]
        terminal = links[-1].build(mesh)
        if terminal.out_field is None:
            raise ParametrizationError(
                f"terminal link {type(links[-1]).__name__} produced a bound link without "
                "out_field; a terminal link must declare the field the physics consumes"
            )
        out_field = terminal.out_field
        return BoundChain(mesh, middle, terminal, out_field, self)


class BoundChain:
    """A chain bound to a mesh; the object the orchestration layer runs."""

    def __init__(
        self,
        mesh: Mesh,
        middle: list[BoundLink],
        terminal: BoundLink,
        out_field: FieldSpec,
        spec: Chain,
    ) -> None:
        self.mesh = mesh
        self.spec = spec
        self._middle = middle
        self._terminal = terminal
        self.out_field = out_field
        self.n_vars = int(mesh.design.sum())
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
        # Clip the absorbed noise instead of passing it on: a value a hair below
        # zero makes a non-integer SIMP exponent produce NaN (a negative base
        # raised to 2.5), and that NaN reaches the stiffness matrix.
        return np.clip(arr, 0.0, 1.0)

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
        rho = self._embed(x)
        for link in self._middle:
            inputs.append(rho)
            rho = link.apply(rho)
        pinned = self._pin(rho)
        inputs.append(pinned)
        return inputs, pinned

    def apply(self, x: Any) -> _F64:
        """Map design variables to the physics field, shape ``(n_elements,)``."""
        return self.evaluate(x).field.copy()

    def physical_density(self, x: Any) -> _F64:
        """Return the physical density field, ``(n_elements,)``, after filters and projection.

        This is the field before the terminal material-interpolation link
        (solid pinned to 1, void to 0). Responses defined on density, such
        as volume, consume it; responses on the interpolated property, such
        as compliance, consume :meth:`apply`. It is also the branch point
        for coupled physics (one terminal link per physics share it).
        """
        return self.evaluate(x).density.copy()

    def pullback(self, x: Any, grad_field: Any) -> _F64:
        """Chain-rule ``dF/d(field)`` back to ``dF/dx`` through the whole chain."""
        return self.evaluate(x).pullback(grad_field)

    def pullback_density(self, x: Any, grad_density: Any) -> _F64:
        """Chain-rule ``dF/d(rho_bar)`` back to ``dF/dx`` for density responses.

        Stops at the terminal boundary: use this for a response defined on
        :meth:`physical_density` (e.g. volume), :meth:`pullback` for one on
        :meth:`apply`.
        """
        return self.evaluate(x).pullback_density(grad_density)

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
        return g[self.mesh.design]

    def _pin_pullback(self, g: _F64) -> _F64:
        out = g.copy()
        out[self.mesh.solid] = 0.0
        out[self.mesh.void] = 0.0
        return out

    def evaluate(self, x: Any) -> ChainEval:
        """Run the forward pass once and return a reusable evaluation.

        This is the one forward pass; :meth:`apply`,
        :meth:`physical_density`, :meth:`pullback` and
        :meth:`pullback_density` are one-line wrappers over it. Call it
        directly when a loop needs several of those per design point (the
        orchestration layer does) and pay the filter pass once instead of
        four times.
        """
        arr = self._check(x)
        inputs, pinned = self._forward(arr)
        return ChainEval(self, inputs, pinned)


class ChainEval:
    """One forward pass shared by field/density reads and their pullbacks.

    Caches the forward arrays only; the links resolve their kernels per
    pullback call, so ``use_backend`` still governs every computation.
    ``density`` and ``field`` are read-only views; copy them before mutating.
    """

    def __init__(self, chain: BoundChain, inputs: list[_F64], pinned: _F64) -> None:
        pinned.setflags(write=False)
        self._chain = chain
        self._inputs = inputs
        self.density = pinned
        self.field: _F64 = chain._terminal.apply(pinned)
        self.field.setflags(write=False)

    def pullback(self, grad_field: Any) -> _F64:
        """Chain-rule ``dF/d(field)`` back to ``dF/dx`` using the cached forward."""
        chain = self._chain
        g = chain._terminal.pullback(self.density, chain._check_grad(grad_field))
        return chain._density_pullback(self._inputs, g)

    def pullback_density(self, grad_density: Any) -> _F64:
        """Chain-rule ``dF/d(rho_bar)`` back to ``dF/dx`` using the cached forward."""
        chain = self._chain
        return chain._density_pullback(self._inputs, chain._check_grad(grad_density))


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
    axis-aligned features. Use :class:`RadialDensityFilter` when isotropic
    reach or a literature-faithful radius-to-member-size relation matters.

    On a non-grid mesh the filter falls back to an explicit sparse-matrix
    implementation with the radial kernel, so ``DensityFilter`` and
    ``RadialDensityFilter`` coincide there; ``radius`` is a physical length
    on every mesh type.
    """

    radius: float = 1.5

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ParametrizationError(f"radius must be > 0, got {self.radius}")

    def build(self, mesh: Mesh) -> BoundLink:
        """Precompute kernels and the mask normalization."""
        if isinstance(mesh, StructuredGrid):
            return _BoundDensityFilter(mesh, self.radius)
        return _BoundMatrixFilter(mesh, self.radius)

    @classmethod
    def fd_example(cls, mesh: Mesh) -> DensityFilter:
        """FD meta-test instance."""
        return cls(radius=1.6 * float(np.max(mesh.element_volumes) ** (1.0 / mesh.dim)))


class _BoundDensityFilter(BoundLink):
    def __init__(self, mesh: StructuredGrid, radius: float) -> None:
        self.mesh = mesh
        self._weights: list[_F64] = []
        halves = _axis_halves(mesh, radius)
        for axis, half in enumerate(halves):
            h = mesh.spacing[axis]
            offs = np.arange(-half, half + 1)
            w = np.maximum(0.0, 1.0 - np.abs(offs) * h / radius)
            self._weights.append(w)
        _warn_if_unfiltered(radius, _unfiltered_axes(mesh, halves))
        active = mesh.active_elements.astype(np.float64)
        self._active = active
        # Bakes in the bind-context kernel (unlike apply/pullback, which
        # resolve per call); fine under the kernels' rtol<=1e-12 agreement
        # contract.
        denom = self._kern()(mesh.to_grid(active), self._weights)
        self._denom = np.maximum(mesh.to_flat(denom), 1e-300)

    def _kern(self) -> Any:
        # resolved per call: use_backend must apply to chains bound outside it
        return get_kernel("separable_correlate")

    def apply(self, x: _F64) -> _F64:
        mesh = self.mesh
        num = self._kern()(mesh.to_grid(x * self._active), self._weights)
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
        back = self._kern()(mesh.to_grid(scaled), self._weights)
        return mesh.to_flat(back) * self._active


@dataclass(frozen=True)
class RadialDensityFilter(LinkSpec):
    """Radial-cone density filter; the topology-optimization literature standard.

    Weights ``max(0, radius - dist)`` over the Euclidean element-center
    distance, mask-aware (void excluded, solid pinned at 1) and row-normalized.
    Unlike :class:`DensityFilter` (a cheaper separable approximation) the reach
    is isotropic, reproducing the 88-line density filter (``ft = 2``); use it
    when matching published references. Cost is O(n*r^dim) vs the separable
    O(n*r), traded for a literature-faithful, orientation-independent kernel.

    On a non-grid mesh this filter and :class:`DensityFilter` both dispatch
    to the same sparse-matrix radial kernel, so they coincide there;
    ``radius`` is a physical length on every mesh type.
    """

    radius: float = 1.5

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ParametrizationError(f"radius must be > 0, got {self.radius}")

    def build(self, mesh: Mesh) -> BoundLink:
        """Precompute the radial kernel and the mask normalization."""
        if isinstance(mesh, StructuredGrid):
            return _BoundRadialDensityFilter(mesh, self.radius)
        return _BoundMatrixFilter(mesh, self.radius)

    @classmethod
    def fd_example(cls, mesh: Mesh) -> RadialDensityFilter:
        """FD meta-test instance."""
        return cls(radius=1.6 * float(np.max(mesh.element_volumes) ** (1.0 / mesh.dim)))


class _BoundRadialDensityFilter(BoundLink):
    def __init__(self, mesh: StructuredGrid, radius: float) -> None:
        self.mesh = mesh
        axes = []
        halves = _axis_halves(mesh, radius)
        for axis, half in enumerate(halves):
            axes.append(np.arange(-half, half + 1) * mesh.spacing[axis])
        _warn_if_unfiltered(radius, _unfiltered_axes(mesh, halves))
        offsets = np.meshgrid(*axes, indexing="ij")
        dist = np.sqrt(sum(o**2 for o in offsets))
        self._kernel = np.maximum(0.0, radius - dist)
        active = mesh.active_elements.astype(np.float64)
        self._active = active
        # Bakes in the bind-context kernel (unlike apply/pullback, which
        # resolve per call); fine under the kernels' rtol<=1e-12 agreement
        # contract.
        denom = self._kern()(mesh.to_grid(active), self._kernel)
        self._denom = np.maximum(mesh.to_flat(denom), 1e-300)

    def _kern(self) -> Any:
        # resolved per call: use_backend must apply to chains bound outside it
        return get_kernel("radial_correlate")

    def apply(self, x: _F64) -> _F64:
        mesh = self.mesh
        num = self._kern()(mesh.to_grid(x * self._active), self._kernel)
        out = mesh.to_flat(num) / self._denom
        return np.where(mesh.active_elements, out, 0.0)

    def pullback(self, x: _F64, grad_out: _F64) -> _F64:
        # forward is A D^-1 C A with the radial correlation C symmetric; the
        # transpose is A C D^-1 A (diagonal factors commute), as below.
        mesh = self.mesh
        scaled = np.where(mesh.active_elements, grad_out / self._denom, 0.0)
        back = self._kern()(mesh.to_grid(scaled), self._kernel)
        return mesh.to_flat(back) * self._active


class _BoundMatrixFilter(BoundLink):
    """Radial hat-kernel filter as an explicit sparse matrix; serves any mesh.

    Weights ``max(0, radius - dist)`` over Euclidean element-centroid
    distance, scaled by neighbor element volume (constant volumes cancel in
    the normalization, so on a uniform mesh this reproduces the radial grid
    filter exactly). Mask semantics match the grid filters: void is excluded,
    normalization runs over active neighbors only.
    """

    def __init__(self, mesh: Mesh, radius: float) -> None:
        from scipy import sparse
        from scipy.spatial import cKDTree

        self.mesh = mesh
        active = np.asarray(mesh.active_elements)
        self._active_mask = active
        self._active = active.astype(np.float64)
        centroids = np.asarray(mesh.element_centroids)
        volumes = np.asarray(mesh.element_volumes)
        idx = np.flatnonzero(active)
        tree = cKDTree(centroids[idx])
        neighbors = tree.query_ball_point(centroids[idx], r=radius)
        counts = np.fromiter((len(nb) for nb in neighbors), dtype=np.int64, count=idx.size)  # type: ignore[arg-type]
        if int(counts.max()) <= 1:
            # every active element neighbors only itself: the same degenerate
            # identity the grid filters warn about, caught here on real distances
            typical = float(np.median(volumes[idx]) ** (1.0 / mesh.dim))
            _warn_if_unfiltered(radius, f"any direction (typical element size {typical:g})")
        rows = np.repeat(idx, counts)
        # active is never empty (an all-void mesh fails at construction) and
        # every point neighbors itself, so the concatenation is never empty
        cols = idx[np.concatenate(neighbors).astype(np.int64)]
        dist = np.linalg.norm(centroids[rows] - centroids[cols], axis=1)
        weights = np.maximum(0.0, radius - dist) * volumes[cols]
        n = mesh.n_elements
        self._w = sparse.csr_matrix((weights, (rows, cols)), shape=(n, n))
        # query_ball_point is inclusive at r=radius, so exact-radius neighbors
        # (weight = radius - dist = 0) get stored as explicit zeros; drop them.
        self._w.eliminate_zeros()
        denom = np.asarray(self._w @ self._active).ravel()
        # every active row has at least its self-weight radius * volume > 0
        self._denom = np.maximum(denom, 1e-300)

    def apply(self, x: _F64) -> _F64:
        num = np.asarray(self._w @ (x * self._active)).ravel()
        return np.where(self._active_mask, num / self._denom, 0.0)

    def pullback(self, x: _F64, grad_out: _F64) -> _F64:
        # forward is A D^-1 W A (mask, normalize, weight, mask); the exact
        # transpose is A W^T D^-1 A. W is asymmetric under volume weighting,
        # so the transpose is real, not cosmetic.
        scaled = np.where(self._active_mask, grad_out / self._denom, 0.0)
        return np.asarray(self._w.T @ scaled).ravel() * self._active


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

    def build(self, mesh: Mesh) -> _BoundHeaviside:
        """Bind; the projection is elementwise and mesh-independent."""
        return _BoundHeaviside(self.beta, self.eta)

    @classmethod
    def fd_example(cls, mesh: Mesh) -> Heaviside:
        """FD meta-test instance."""
        return cls(beta=3.0)


class _BoundHeaviside(BoundLink):
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

    def build(self, mesh: Mesh) -> _BoundSIMP:
        """Bind; the interpolation is elementwise."""
        return _BoundSIMP(self.p, self.scale_min)

    @classmethod
    def fd_example(cls, mesh: Mesh) -> SIMP:
        """FD meta-test instance."""
        return cls(p=3.0)


class _BoundSIMP(BoundLink):
    out_field = STIFFNESS_SCALE

    def __init__(self, p: float, scale_min: float) -> None:
        self.p = p
        self.scale_min = scale_min

    def apply(self, x: _F64) -> _F64:
        return np.asarray(self.scale_min + x**self.p * (1.0 - self.scale_min))

    def pullback(self, x: _F64, grad_out: _F64) -> _F64:
        return np.asarray(grad_out * self.p * x ** (self.p - 1.0) * (1.0 - self.scale_min))
