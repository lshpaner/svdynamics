import numpy as np
import pytest

from svdynamics import AdaptiveCompositeKernel, CompositeKernel

KERNELS = ["linear", ("rbf", {"gamma": 1.0}), ("poly", {"degree": 2})]


@pytest.mark.parametrize("weights", [[-0.1, 0.6, 0.5], [1.0, -1.0, 1.0]])
def test_negative_weights_rejected(weights):
    with pytest.raises(ValueError, match="nonnegative"):
        CompositeKernel(KERNELS, weights=weights)


def test_zero_weight_allowed():
    k = CompositeKernel(KERNELS, weights=[0.0, 1.0, 1.0])
    np.testing.assert_allclose(k.normalized_weights, [0.0, 0.5, 0.5])


def test_all_zero_weights_rejected():
    with pytest.raises(ValueError, match="sum to 0"):
        CompositeKernel(KERNELS, weights=[0.0, 0.0, 0.0])


def test_adaptive_initial_negative_weights_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        AdaptiveCompositeKernel(KERNELS, weights=[-1.0, 1.0, 1.0])
