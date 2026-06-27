"""Tests for the continuation Schedule/Stage types."""

import pytest

from topokit.problem import ProblemError, Schedule, Stage


def test_default_schedule_ramps_p_then_beta() -> None:
    s = Schedule.default()
    pairs = [(st.p, st.beta) for st in s.stages]
    assert pairs == [(1, 1), (2, 1), (3, 1), (3, 2), (3, 4), (3, 8), (3, 16), (3, 32)]
    assert all(st.max_iter == 200 and st.tol == 0.01 for st in s.stages)


def test_single_schedule_one_stage() -> None:
    s = Schedule.single(p=3.0, beta=1.0, max_iter=120, tol=1e-3)
    assert len(s.stages) == 1
    assert s.stages[0].p == 3.0 and s.stages[0].max_iter == 120


def test_stage_and_schedule_validation() -> None:
    with pytest.raises(ProblemError, match="p"):
        Stage(p=0.5, beta=1.0)
    with pytest.raises(ProblemError, match="beta"):
        Stage(p=3.0, beta=0.0)
    with pytest.raises(ProblemError, match="max_iter"):
        Stage(p=3.0, beta=1.0, max_iter=0)
    with pytest.raises(ProblemError, match="tol"):
        Stage(p=3.0, beta=1.0, tol=-1.0)
    with pytest.raises(ProblemError, match="stages"):
        Schedule(())
