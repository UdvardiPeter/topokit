# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Tests for typed field containers."""

from pathlib import Path

import numpy as np
import pytest

from topokit.fields import DesignField, ElementField, FieldError, NodeField
from topokit.mesh import StructuredGrid

# a real mesh, since the field containers now take the Mesh protocol:
# a 3x2 grid is 6 elements and 12 nodes, which is what these shapes assume
MESH = StructuredGrid(shape=(3, 2), spacing=(1.0, 1.0))


def test_design_field_validates_length() -> None:
    with pytest.raises(FieldError, match="6"):
        DesignField(np.zeros(5), MESH, name="rho")


def test_values_coerced_to_float64_copy() -> None:
    raw = np.ones(6, dtype=np.float32)
    f = DesignField(raw, MESH, name="rho")
    assert f.values.dtype == np.float64
    raw[0] = 99.0
    assert f.values[0] == 1.0


def test_values_are_read_only() -> None:
    f = DesignField(np.zeros(6), MESH, name="rho")
    with pytest.raises(ValueError, match="read-only"):
        f.values[0] = 1.0


def test_node_field_shape_is_nodes_by_components() -> None:
    f = NodeField(np.zeros((12, 2)), MESH, name="u")
    assert f.values.shape == (12, 2)
    with pytest.raises(FieldError):
        NodeField(np.zeros((11, 2)), MESH, name="u")


def test_element_field_roundtrip_save_load(tmp_path: Path) -> None:
    f = ElementField(np.arange(6, dtype=float), MESH, name="von_mises")
    p = tmp_path / "vm.npz"
    f.save(p)
    g = ElementField.load(p, MESH)
    assert g.name == "von_mises"
    np.testing.assert_array_equal(g.values, f.values)


def test_load_rejects_mismatched_mesh(tmp_path: Path) -> None:
    f = DesignField(np.zeros(6), MESH, name="rho")
    p = tmp_path / "rho.npz"
    f.save(p)
    with pytest.raises(FieldError, match="mesh"):
        DesignField.load(p, StructuredGrid(shape=(2, 1), spacing=(1.0, 1.0)))


def test_field_spec_compares_by_value() -> None:
    from topokit.fields import FieldSpec

    assert FieldSpec("stiffness_scale") == FieldSpec("stiffness_scale")
    assert FieldSpec("stiffness_scale") != FieldSpec("conductivity_scale")
