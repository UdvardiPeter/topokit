"""Self-tests of the finite-difference gradient harness."""

import numpy as np
import numpy.typing as npt
import pytest

from topokit.testing import GradientMismatchError, assert_gradient_matches

pytestmark = pytest.mark.fd

RNG = np.random.default_rng(seed=20260609)


def quadratic(x: npt.NDArray[np.float64]) -> float:
    return float(x @ x + 3.0 * x.sum())


def quadratic_grad(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return 2.0 * x + 3.0


def test_correct_gradient_passes() -> None:
    x = RNG.normal(size=20)
    assert_gradient_matches(quadratic, quadratic_grad, x)


def test_wrong_gradient_fails_with_diagnostics() -> None:
    x = RNG.normal(size=20)

    def bad_grad(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return 2.0 * x  # missing the +3 term

    with pytest.raises(GradientMismatchError, match="rel error"):
        assert_gradient_matches(quadratic, bad_grad, x)


def test_poorly_scaled_input_still_verifies() -> None:
    x = RNG.normal(size=20) * 1e3
    assert_gradient_matches(quadratic, quadratic_grad, x)
