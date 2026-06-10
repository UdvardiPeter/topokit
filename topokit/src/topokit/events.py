# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Peter Udvardi and TopoKit contributors
"""Typed, synchronous event bus for the optimization loop.

Subscriber exceptions are caught and logged. Payloads that reference
higher-layer types are typed as ``object``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import cast

_log = logging.getLogger("topokit.events")


@dataclass(frozen=True, slots=True)
class Event:
    """Base class for all TopoKit events."""


@dataclass(frozen=True, slots=True)
class StudyStarted(Event):
    """Published once when a study run begins."""

    config: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class IterationFinished(Event):
    """Published after every optimization iteration."""

    iteration: int
    design_change: float
    responses: Mapping[str, float]
    wall_time: float


@dataclass(frozen=True, slots=True)
class FieldSnapshot(Event):
    """Periodic snapshot of the design field."""

    iteration: int
    rho: object


@dataclass(frozen=True, slots=True)
class StageFinished(Event):
    """Published when a continuation stage converges or hits its cap."""

    stage: int
    reason: str


@dataclass(frozen=True, slots=True)
class CheckResult(Event):
    """Result of a manufacturability design check."""

    name: str
    passed: bool
    details: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StudyFinished(Event):
    """Published once when a study run ends."""

    reason: str
    summary: Mapping[str, object]


class EventBus:
    """Dispatches events synchronously to type-matched subscribers."""

    def __init__(self) -> None:
        self._subs: list[tuple[type[Event], Callable[[Event], None]]] = []

    def subscribe[E: Event](
        self, event_type: type[E], fn: Callable[[E], None]
    ) -> Callable[[], None]:
        """Call ``fn`` for events of ``event_type`` or its subclasses.

        Returns a function that removes the subscription.
        """
        entry = (event_type, cast(Callable[[Event], None], fn))
        self._subs.append(entry)

        def unsubscribe() -> None:
            if entry in self._subs:
                self._subs.remove(entry)

        return unsubscribe

    def publish(self, event: Event) -> None:
        """Send ``event`` to all matching subscribers.

        Subscriber exceptions are logged and do not stop dispatch.
        """
        for event_type, fn in list(self._subs):
            if isinstance(event, event_type):
                try:
                    fn(event)
                except Exception:
                    _log.exception("event subscriber %r failed on %s", fn, type(event).__name__)
