# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Runnable docs stay runnable: Home, tutorial, and extend pages execute as written."""

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[2] / "docs" / "content"
PAGES = (
    [DOCS / "index.md"]  # noqa: RUF005 (extend/ is empty until a later task; keep the append shape)
    + sorted((DOCS / "tutorials").glob("*.md"))
    + sorted((DOCS / "extend").glob("*.md"))
)

_FENCE = re.compile(r"(<!--\s*no-run\s*-->\s*\n)?```python\n(.*?)```", re.DOTALL)


def _runnable_blocks(page: Path) -> str:
    matches = _FENCE.finditer(page.read_text())
    return "\n\n".join(m.group(2) for m in matches if not m.group(1))


@pytest.mark.regression_full
@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_docs_page_runs(page: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code = _runnable_blocks(page)
    if not code.strip():
        pytest.skip(f"{page.name} has no executable code")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    scope: dict[str, object] = {}
    exec(compile(code, str(page), "exec"), scope)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pass
    else:
        plt.close("all")
