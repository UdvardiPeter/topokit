# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""``.topo`` checkpoint serialization: a zip of ``manifest.json`` + ``arrays.npz``.

Pure functions over plain dicts and arrays; the orchestration layer builds the
manifest and reads it back. State-only (no Problem reconstruction -- that is
WP-4.1's ``problem.json``); a config fingerprint guards against resuming the
wrong problem.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from typing import Any

import numpy as np
import numpy.typing as npt

_F64 = npt.NDArray[np.float64]
SCHEMA_VERSION = 1


def config_fingerprint(parts: tuple[Any, ...]) -> str:
    """Return a deterministic sha256 over a canonical view of the problem's structure."""
    return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()


def write_topo(path: str, manifest: dict[str, Any], arrays: dict[str, _F64]) -> None:
    """Write a ``.topo`` atomically (temp file + ``os.replace``)."""
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    tmp = f"{path}.tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
        zf.writestr("arrays.npz", buf.getvalue())
    os.replace(tmp, path)


def read_topo(path: str) -> tuple[dict[str, Any], dict[str, _F64]]:
    """Read back the manifest dict and the arrays dict."""
    with zipfile.ZipFile(path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))
        with io.BytesIO(zf.read("arrays.npz")) as b:
            npz = np.load(b)
            arrays = {k: npz[k] for k in npz.files}
    return manifest, arrays
