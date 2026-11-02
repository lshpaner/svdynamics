import numpy as np
import pytest
from sklearn.datasets import make_circles

from svdynamics import AdaptiveCompositeKernel
from svdynamics.kernels import (
    _alignment_weights,
    _kernel_target_alignment,
)


def test_kernel_target_alignment_perfect():
    y = np.array([-1, -1, 1, 1])
    target_kernel = np.outer(y, y)

    alignment = _kernel_target_alignment(
        target_kernel,
        y,
    )

    assert np.isclose(alignment, 1.0)


def test_kernel_target_alignment_opposite():
    y = np.array([-1, -1, 1, 1])
    target_kernel = np.outer(y, y)

    alignment = _kernel_target_alignment(
        -target_kernel,
        y,
    )

    assert np.isclose(alignment, -1.0)


def test_kernel_target_alignment_zero_kernel():
    y = np.array([-1, -1, 1, 1])
    zero_kernel = np.zeros((4, 4))

    alignment = _kernel_target_alignment(
        zero_kernel,
        y,
    )

    assert np.isclose(alignment, 0.0)


def test_alignment_weights_sum_to_one_and_are_nonnegative():
    y = np.array([-1, -1, 1, 1])
    target_kernel = np.outer(y, y)

    kernels = [
        target_kernel,
        0.5 * target_kernel,
        -target_kernel,
    ]

    weights = _alignment_weights(
        kernels,
        y,
    )

    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0.0)


def test_alignment_weights_uniform_fallback():
    y = np.array([-1, -1, 1, 1])
    kernels = [
        np.zeros((4, 4)),
        np.zeros((4, 4)),
        np.zeros((4, 4)),
    ]

    weights = _alignment_weights(
        kernels,
        y,
    )

    assert np.allclose(
        weights,
        np.full(3, 1.0 / 3.0),
    )


def test_adaptive_composite_kernel_fit_stores_diagnostics():
    X, y = make_circles(
        n_samples=200,
        factor=0.4,
        noise=0.05,
        random_state=42,
    )
    y = 2 * y - 1

    kernel = AdaptiveCompositeKernel(
        kernels=[
            "linear",
            ("rbf", {"gamma": 1.0}),
            ("poly", {"degree": 2}),
        ],
    )

    returned = kernel.fit(X, y)

    assert returned is kernel
    assert kernel.alignments_.shape == (3,)
    assert kernel.weights_.shape == (3,)
    assert np.all(np.isfinite(kernel.alignments_))
    assert np.all(np.isfinite(kernel.weights_))
    assert np.all(kernel.weights_ >= 0.0)
    assert np.isclose(kernel.weights_.sum(), 1.0)

    # On concentric circles, RBF should receive the largest weight.
    assert np.argmax(kernel.weights_) == 1


def test_adaptive_composite_kernel_fit_rejects_length_mismatch():
    X = np.zeros((5, 2))
    y = np.array([-1, 1, -1, 1])

    kernel = AdaptiveCompositeKernel(
        kernels=["linear", "rbf"],
    )

    with pytest.raises(ValueError):
        kernel.fit(X, y)
