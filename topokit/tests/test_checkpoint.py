# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Tests for the .topo checkpoint serialization."""

from pathlib import Path

import numpy as np

from topokit.checkpoint import config_fingerprint, read_topo, write_topo


def test_topo_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "run.topo"
    manifest = {"schema": 1, "iteration": 5, "fingerprint": "abc"}
    arrays = {"x": np.arange(4.0), "best_x": np.ones(3)}
    write_topo(str(path), manifest, arrays)
    m, a = read_topo(str(path))
    assert m["iteration"] == 5
    np.testing.assert_array_equal(a["x"], np.arange(4.0))


def test_fingerprint_is_deterministic_and_sensitive() -> None:
    a = config_fingerprint(("mesh", (20, 10), "simp", 3.0))
    b = config_fingerprint(("mesh", (20, 10), "simp", 3.0))
    c = config_fingerprint(("mesh", (20, 10), "simp", 4.0))
    assert a == b and a != c


def test_atomic_write_leaves_no_partial(tmp_path: Path) -> None:
    path = tmp_path / "run.topo"
    write_topo(str(path), {"schema": 1}, {"x": np.zeros(2)})
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp"))
