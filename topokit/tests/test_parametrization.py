# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Tests for the unified parametrization chain."""

import numpy as np
import pytest

from topokit.fields import FieldSpec
from topokit.mesh import StructuredGrid
from topokit.parametrization import (
    SIMP,
    Chain,
    DensityFilter,
    Heaviside,
    ParametrizationError,
    RadialDensityFilter,
    SensitivityFilter,
    SymmetryMap,
)
from topokit.testing import assert_gradient_matches

G42 = StructuredGrid(shape=(4, 2), spacing=(1.0, 1.0))


def test_pipe_composition_builds_chain() -> None:
    chain = SymmetryMap(planes=("x",)) | DensityFilter(radius=1.5) | Heaviside() | SIMP()
    assert isinstance(chain, Chain)
    assert len(chain.links) == 4


def test_bind_requires_terminal_link() -> None:
    with pytest.raises(ParametrizationError, match="terminal"):
        (DensityFilter(radius=1.5) | Heaviside()).bind(G42)


def test_terminal_must_be_last_and_unique() -> None:
    with pytest.raises(ParametrizationError, match="terminal"):
        (SIMP() | Heaviside()).bind(G42)
    with pytest.raises(ParametrizationError, match="terminal"):
        (SIMP() | SIMP()).bind(G42)


def test_symmetry_only_first() -> None:
    with pytest.raises(ParametrizationError, match="first"):
        (DensityFilter(radius=1.5) | SymmetryMap(planes=("x",)) | SIMP()).bind(G42)


def test_both_filters_warn() -> None:
    with pytest.warns(UserWarning, match="SensitivityFilter"):
        (DensityFilter(radius=1.5) | SensitivityFilter(radius=1.5) | SIMP()).bind(G42)


def test_out_field_and_sizes() -> None:
    bound = (Heaviside() | SIMP()).bind(G42)
    assert bound.out_field == FieldSpec("stiffness_scale")
    assert bound.n_vars == 8
    scale = bound.apply(np.full(8, 0.4))
    assert scale.shape == (8,)


def test_initial_design() -> None:
    bound = SIMP().bind(G42)
    x0 = bound.initial_design(0.3)
    assert x0.shape == (8,)
    np.testing.assert_allclose(x0, 0.3)


def test_simp_values_and_floor() -> None:
    bound = SIMP(p=3.0, scale_min=1e-9).bind(G42)
    rho = np.full(8, 0.5)
    scale = bound.apply(rho)
    np.testing.assert_allclose(scale, 1e-9 + 0.5**3 * (1 - 1e-9), rtol=1e-12)
    zero = bound.apply(np.zeros(8))
    np.testing.assert_allclose(zero, 1e-9)


def test_heaviside_endpoints_and_midpoint() -> None:
    bound = (Heaviside(beta=4.0, eta=0.5) | SIMP(p=1.0)).bind(G42)
    rho = np.array([0.0, 1.0, 0.5, 0.2, 0.8, 0.0, 1.0, 0.5])
    # SIMP(p=1) is affine, so projection values survive up to the tiny floor
    out = bound.apply(rho)
    assert out[0] == pytest.approx(0.0, abs=1e-8)
    assert out[1] == pytest.approx(1.0, abs=1e-8)
    assert out[2] == pytest.approx(0.5, abs=1e-8)
    assert out[3] < rho[3]  # sharpened below eta
    assert out[4] > rho[4]  # sharpened above eta


def test_density_filter_hand_values() -> None:
    g = StructuredGrid(shape=(3, 1), spacing=(1.0, 1.0))
    bound = (DensityFilter(radius=1.1) | SIMP(p=1.0, scale_min=0.0)).bind(g)
    out = bound.apply(np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(out, [11.0 / 12.0, 1.0 / 13.0, 0.0], rtol=1e-12)


def test_density_filter_uniform_invariance_masked_anisotropic() -> None:
    void = np.zeros(12, dtype=bool)
    void[[5, 6]] = True
    g = StructuredGrid(shape=(4, 3), spacing=(0.7, 0.45), void=void)
    bound = (DensityFilter(radius=1.0) | SIMP(p=1.0, scale_min=0.0)).bind(g)
    x = np.full(bound.n_vars, 0.6)
    out = bound.apply(x)
    # solid/void pinning aside, a uniform active field stays uniform
    np.testing.assert_allclose(out[g.design], 0.6, rtol=1e-12)
    np.testing.assert_allclose(out[g.void], 0.0)


def test_density_filter_solid_pulls_neighbors_up() -> None:
    solid = np.zeros(8, dtype=bool)
    solid[0] = True
    g = StructuredGrid(shape=(4, 2), spacing=(1.0, 1.0), solid=solid)
    bound = (DensityFilter(radius=1.5) | SIMP(p=1.0, scale_min=0.0)).bind(g)
    out = bound.apply(np.zeros(bound.n_vars))
    assert out[0] == pytest.approx(1.0)  # solid pinned
    assert out[1] > 0.0  # influenced by the solid neighbor
    assert out[3] == pytest.approx(0.0)  # out of filter reach


def test_radial_filter_reduces_to_separable_in_1d() -> None:
    # in 1D the radial and separable kernels are proportional, so identical
    # after the row normalization
    g = StructuredGrid(shape=(3, 1), spacing=(1.0, 1.0))
    x = np.array([1.0, 0.0, 0.0])
    sep = (DensityFilter(radius=1.1) | SIMP(p=1.0, scale_min=0.0)).bind(g).apply(x)
    rad = (RadialDensityFilter(radius=1.1) | SIMP(p=1.0, scale_min=0.0)).bind(g).apply(x)
    np.testing.assert_allclose(rad, sep, rtol=1e-12)


def test_radial_differs_from_separable_in_2d() -> None:
    # in 2D the radial cone weights diagonal neighbors less than the separable box
    g = StructuredGrid(shape=(3, 3), spacing=(1.0, 1.0))
    impulse = np.zeros(9)
    impulse[4] = 1.0  # center element (x-fastest id 4 = (1, 1))
    sep = (DensityFilter(radius=1.5) | SIMP(p=1.0, scale_min=0.0)).bind(g).apply(impulse)
    rad = (RadialDensityFilter(radius=1.5) | SIMP(p=1.0, scale_min=0.0)).bind(g).apply(impulse)
    assert not np.allclose(rad, sep)
    gs, gr = g.to_grid(sep), g.to_grid(rad)
    # diagonal-to-face response ratio is smaller under the radial cone
    assert gr[0, 0] / gr[1, 0] < gs[0, 0] / gs[1, 0]


def test_radial_uniform_invariance() -> None:
    g = StructuredGrid(shape=(4, 3), spacing=(0.7, 0.45))
    bound = (RadialDensityFilter(radius=1.2) | SIMP(p=1.0, scale_min=0.0)).bind(g)
    np.testing.assert_allclose(bound.apply(np.full(bound.n_vars, 0.6)), 0.6, rtol=1e-12)


def test_symmetry_map_explicit_orbits() -> None:
    bound = (SymmetryMap(planes=("x",)) | SIMP(p=1.0, scale_min=0.0)).bind(G42)
    assert bound.n_vars == 4
    x = np.array([0.1, 0.2, 0.3, 0.4])
    out = bound.apply(x)
    grid = G42.to_grid(out)
    np.testing.assert_allclose(grid[0, :], grid[3, :])
    np.testing.assert_allclose(grid[1, :], grid[2, :])


def test_symmetry_requires_symmetric_masks() -> None:
    void = np.zeros(8, dtype=bool)
    void[0] = True  # not mirror-symmetric
    g = StructuredGrid(shape=(4, 2), spacing=(1.0, 1.0), void=void)
    with pytest.raises(ParametrizationError, match="symmetric"):
        (SymmetryMap(planes=("x",)) | SIMP()).bind(g)


def test_sensitivity_filter_identity_apply() -> None:
    bound = (SensitivityFilter(radius=1.5) | SIMP(p=1.0, scale_min=0.0)).bind(G42)
    rho = np.linspace(0.1, 0.8, 8)
    np.testing.assert_allclose(bound.apply(rho), rho, rtol=1e-12)


def test_chain_validates_input() -> None:
    bound = SIMP().bind(G42)
    with pytest.raises(ParametrizationError, match="shape"):
        bound.apply(np.ones(5))
    bad = np.full(8, 0.5)
    bad[0] = np.nan
    with pytest.raises(ParametrizationError, match="finite"):
        bound.apply(bad)


def test_full_chain_fd_anisotropic_masked() -> None:
    void = np.zeros(12, dtype=bool)
    void[[1, 2]] = True  # symmetric about the x mirror (columns 1, 2 of row 0)
    solid = np.zeros(12, dtype=bool)
    solid[[9, 10]] = True
    g = StructuredGrid(shape=(4, 3), spacing=(0.7, 0.45), void=void, solid=solid)
    chain = (
        SymmetryMap(planes=("x",)) | DensityFilter(radius=1.0) | Heaviside(beta=2.0) | SIMP(p=3.0)
    )
    bound = chain.bind(g)
    rng = np.random.default_rng(seed=20260612)
    x = rng.uniform(0.2, 0.8, size=bound.n_vars)
    v = rng.normal(size=g.n_elements)

    def f(xx: np.ndarray) -> float:
        return float(v @ bound.apply(xx))

    def grad(xx: np.ndarray) -> np.ndarray:
        return np.asarray(bound.pullback(xx, v))

    assert_gradient_matches(f, grad, x)


def test_registry_links_fd_verified_or_exempt() -> None:
    from topokit.registry import registry

    g = StructuredGrid(shape=(4, 2), spacing=(0.7, 0.45))
    rng = np.random.default_rng(seed=7)
    for name in registry.names("chain_links"):
        cls = registry.get("chain_links", name)
        if cls.fd_exempt is not None:
            assert isinstance(cls.fd_exempt, str) and cls.fd_exempt
            continue
        link = cls.fd_example(g)
        chain = Chain((link,)) if link.is_terminal else Chain((link, SIMP()))
        bound = chain.bind(g)
        x = rng.uniform(0.2, 0.8, size=bound.n_vars)
        v = rng.normal(size=g.n_elements)

        def f(xx: np.ndarray, b: object = bound, vv: np.ndarray = v) -> float:
            return float(vv @ b.apply(xx))  # type: ignore[attr-defined]

        def grad(xx: np.ndarray, b: object = bound, vv: np.ndarray = v) -> np.ndarray:
            return np.asarray(b.pullback(xx, vv))  # type: ignore[attr-defined]

        assert_gradient_matches(f, grad, x)


def test_spec_validation_errors() -> None:
    with pytest.raises(ParametrizationError, match="radius"):
        DensityFilter(radius=0.0)
    with pytest.raises(ParametrizationError, match="beta"):
        Heaviside(beta=-1.0)
    with pytest.raises(ParametrizationError, match="p"):
        SIMP(p=0.0)
    with pytest.raises(ParametrizationError, match="plane"):
        SymmetryMap(planes=("q",))


def test_physical_density_is_pre_terminal() -> None:
    g = StructuredGrid(shape=(4, 2), spacing=(1.0, 1.0))
    bound = (DensityFilter(radius=1.5) | SIMP(p=3.0)).bind(g)
    x = np.full(bound.n_vars, 0.5)
    rho_bar = bound.physical_density(x)
    # rho_bar is the filtered density (~0.5), the scale is rho_bar**3 (~0.125)
    np.testing.assert_allclose(rho_bar, 0.5, rtol=1e-9)
    np.testing.assert_allclose(bound.apply(x), 0.5**3, rtol=1e-6)


def test_physical_density_pins_solid_and_void() -> None:
    solid = np.zeros(8, dtype=bool)
    solid[0] = True
    void = np.zeros(8, dtype=bool)
    void[7] = True
    g = StructuredGrid(shape=(4, 2), spacing=(1.0, 1.0), solid=solid, void=void)
    bound = SIMP().bind(g)
    rho = bound.physical_density(np.full(bound.n_vars, 0.4))
    assert rho[0] == 1.0  # solid
    assert rho[7] == 0.0  # void


def test_pullback_density_fd_verified() -> None:
    void = np.zeros(12, dtype=bool)
    void[[1, 2]] = True
    solid = np.zeros(12, dtype=bool)
    solid[[9, 10]] = True
    g = StructuredGrid(shape=(4, 3), spacing=(0.7, 0.45), void=void, solid=solid)
    bound = (
        SymmetryMap(planes=("x",)) | DensityFilter(radius=1.0) | Heaviside(beta=2.0) | SIMP()
    ).bind(g)
    rng = np.random.default_rng(seed=11)
    x = rng.uniform(0.2, 0.8, size=bound.n_vars)
    w = rng.normal(size=g.n_elements)

    def f(xx: np.ndarray) -> float:
        return float(w @ bound.physical_density(xx))

    def grad(xx: np.ndarray) -> np.ndarray:
        return np.asarray(bound.pullback_density(xx, w))

    assert_gradient_matches(f, grad, x)


def test_out_of_range_design_raises() -> None:
    bound = SIMP(p=3.0).bind(G42)
    with pytest.raises(ParametrizationError, match=r"\[0, 1\]"):
        bound.apply(np.full(8, -0.01))
    with pytest.raises(ParametrizationError, match=r"\[0, 1\]"):
        bound.apply(np.full(8, 1.5))
    # float noise at the bounds is tolerated
    bound.apply(np.full(8, 1.0 + 1e-9))
    bound.apply(np.full(8, -1e-9))


def test_3d_multiplane_symmetry_fd_and_octants() -> None:
    g = StructuredGrid(shape=(4, 4, 4), spacing=(1.0, 1.0, 1.0))
    bound = (SymmetryMap(planes=("x", "y", "z")) | DensityFilter(radius=1.5) | SIMP(p=3.0)).bind(g)
    assert bound.n_vars == 8  # 8 octants of a 4x4x4 grid
    rng = np.random.default_rng(seed=3)
    x = rng.uniform(0.2, 0.8, size=bound.n_vars)
    out = g.to_grid(bound.apply(x))
    np.testing.assert_allclose(out, out[::-1])
    np.testing.assert_allclose(out, out[:, ::-1])
    np.testing.assert_allclose(out, out[:, :, ::-1])
    v = rng.normal(size=g.n_elements)
    assert_gradient_matches(
        lambda xx: float(v @ bound.apply(xx)),
        lambda xx: np.asarray(bound.pullback(xx, v)),
        x,
    )


def test_odd_width_mirror_fixes_center() -> None:
    g = StructuredGrid(shape=(3, 2), spacing=(1.0, 1.0))
    bound = (SymmetryMap(planes=("x",)) | SIMP(p=1.0, scale_min=0.0)).bind(g)
    out = g.to_grid(bound.apply(np.linspace(0.1, 0.9, bound.n_vars)))
    np.testing.assert_allclose(out[0], out[2])  # mirrored columns
    rng = np.random.default_rng(5)
    x = rng.uniform(0.2, 0.8, bound.n_vars)
    v = rng.normal(size=g.n_elements)
    assert_gradient_matches(
        lambda xx: float(v @ bound.apply(xx)),
        lambda xx: np.asarray(bound.pullback(xx, v)),
        x,
    )


def test_heaviside_continuation_endpoint_fd() -> None:
    g = StructuredGrid(shape=(6, 4), spacing=(1.0, 1.0))
    bound = (DensityFilter(radius=1.5) | Heaviside(beta=32.0, eta=0.3) | SIMP(p=3.0)).bind(g)
    rng = np.random.default_rng(9)
    x = rng.uniform(0.2, 0.8, bound.n_vars)
    v = rng.normal(size=g.n_elements)
    assert_gradient_matches(
        lambda xx: float(v @ bound.apply(xx)),
        lambda xx: np.asarray(bound.pullback(xx, v)),
        x,
    )


def test_empty_design_region_raises() -> None:
    g = StructuredGrid(shape=(2, 2), spacing=(1.0, 1.0), solid=np.ones(4, dtype=bool))
    with pytest.raises(ParametrizationError, match="design region is empty"):
        SIMP().bind(g)


def test_simp_requires_p_at_least_one() -> None:
    with pytest.raises(ParametrizationError, match="p must be >= 1"):
        SIMP(p=0.5)
    with pytest.raises(ParametrizationError, match="p must be >= 1"):
        SIMP(p=0.0)
    SIMP(p=1.0)  # linear is allowed


def test_radial_filter_with_sensitivity_filter_warns() -> None:
    with pytest.warns(UserWarning, match="SensitivityFilter"):
        (RadialDensityFilter(radius=1.5) | SensitivityFilter(radius=1.5) | SIMP()).bind(G42)
