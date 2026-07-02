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


def _stage_boundaries(history: dict[str, list[float]]) -> list[int]:
    """Return the iteration indices where a new continuation stage begins."""
    stages = history.get("stage", [])
    return [i for i in range(1, len(stages)) if stages[i] != stages[i - 1]]


def plot_convergence(result: _HasHistory, *, keys: Sequence[str] | None = None) -> Figure:
    """Plot objective, design-change, and constraint/response series vs iteration.

    One subplot per series (small multiples, shared x): the series live on very
    different scales, so a shared axis would flatten all but the largest.
    Continuation-stage boundaries are marked with vertical lines. Pass ``keys``
    to select a subset of ``result.history`` (defaults to all non-bookkeeping
    series). Returns a headless matplotlib ``Figure``.
    """
    require_matplotlib()
    from matplotlib.figure import Figure

    history = result.history
    names = list(keys) if keys is not None else [k for k in history if k not in _BOOKKEEPING]
    if not names:
        fig = Figure()
        fig.subplots()
        return fig
    fig = Figure(figsize=(6.4, max(4.8, 1.6 * len(names))))
    axes = fig.subplots(len(names), 1, sharex=True)
    axes_list = [axes] if len(names) == 1 else list(axes)
    boundaries = _stage_boundaries(history)
    for ax, name in zip(axes_list, names, strict=True):
        series: Any = history.get(name, [])
        ax.plot(range(len(series)), series, label=name)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        for b in boundaries:
            ax.axvline(b, color="0.6", linewidth=0.8, linestyle="--")
    axes_list[-1].set_xlabel("iteration")
    return fig
