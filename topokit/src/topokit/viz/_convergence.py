# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Convergence curves (matplotlib)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

from topokit.viz._backend import require_matplotlib

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# history keys that are run bookkeeping, not curves to plot
_BOOKKEEPING = frozenset({"kkt", "stage"})


class _HasHistory(Protocol):
    history: dict[str, list[float]]


def plot_convergence(result: _HasHistory, *, keys: Sequence[str] | None = None) -> Figure:
    """Plot objective, design-change, and constraint/response series vs iteration.

    Pass ``keys`` to select a subset of ``result.history`` (defaults to all
    non-bookkeeping series). Returns a headless matplotlib ``Figure``.
    """
    require_matplotlib()
    from matplotlib.figure import Figure

    history = result.history
    names = list(keys) if keys is not None else [k for k in history if k not in _BOOKKEEPING]
    fig = Figure()
    ax = fig.subplots()
    for name in names:
        series: Any = history.get(name, [])
        ax.plot(range(len(series)), series, label=name)
    ax.set_xlabel("iteration")
    ax.set_ylabel("value")
    ax.legend()
    return fig
