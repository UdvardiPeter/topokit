# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""The README quickstart runs as written (nightly; docs that rot are worse than none)."""

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[2] / "README.md"


def _quickstart_block() -> str:
    match = re.search(r"```python\n(.*?)```", README.read_text(), re.DOTALL)
    assert match is not None, "README has no python code block"
    return match.group(1)


@pytest.mark.regression_full
def test_readme_quickstart_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # the snippet saves a file
    scope: dict[str, object] = {}
    exec(compile(_quickstart_block(), str(README), "exec"), scope)
    result = scope["result"]
    assert getattr(result, "converged", False), "quickstart run did not converge"
    assert (tmp_path / "cantilever.npz").exists()
