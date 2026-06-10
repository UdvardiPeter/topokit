"""Tests for the event bus."""

import dataclasses
import logging

import pytest

from topokit.events import (
    CheckResult,
    Event,
    EventBus,
    FieldSnapshot,
    IterationFinished,
    StageFinished,
    StudyFinished,
    StudyStarted,
)


def _iter_event(i: int = 1) -> IterationFinished:
    return IterationFinished(
        iteration=i, design_change=0.5, responses={"compliance": 1.0}, wall_time=0.01
    )


def test_subscribe_receives_matching_type_only() -> None:
    bus = EventBus()
    seen: list[Event] = []
    bus.subscribe(IterationFinished, seen.append)
    bus.publish(_iter_event())
    bus.publish(StudyFinished(reason="converged", summary={}))
    assert [type(e) for e in seen] == [IterationFinished]


def test_catch_all_subscription_via_base_event() -> None:
    bus = EventBus()
    seen: list[Event] = []
    bus.subscribe(Event, seen.append)
    bus.publish(StudyStarted(config={}))
    bus.publish(_iter_event())
    assert len(seen) == 2


def test_subscriber_exception_is_logged_not_raised(caplog: pytest.LogCaptureFixture) -> None:
    bus = EventBus()
    seen: list[Event] = []

    def broken(_: Event) -> None:
        raise RuntimeError("viz crashed")

    bus.subscribe(IterationFinished, broken)
    bus.subscribe(IterationFinished, seen.append)
    with caplog.at_level(logging.ERROR, logger="topokit.events"):
        bus.publish(_iter_event())
    assert len(seen) == 1
    assert "viz crashed" in caplog.text


def test_unsubscribe() -> None:
    bus = EventBus()
    seen: list[Event] = []
    unsub = bus.subscribe(IterationFinished, seen.append)
    unsub()
    bus.publish(_iter_event())
    assert seen == []


def test_events_are_frozen() -> None:
    ev = _iter_event()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.iteration = 2  # type: ignore[misc]


def test_all_event_types_constructible() -> None:
    StudyStarted(config={"volume_fraction": 0.3})
    FieldSnapshot(iteration=5, rho=object())
    StageFinished(stage=2, reason="design change below tol")
    CheckResult(name="enclosed_voids", passed=False, details={"count": 1})
    StudyFinished(reason="max iterations", summary={"iterations": 200})
