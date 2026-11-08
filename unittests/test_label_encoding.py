import numpy as np
import pytest
from sklearn.datasets import make_circles

from svdynamics import AdaptiveCompositeKernel

KERNELS = ["linear", ("rbf", {"gamma": 2.0}), ("poly", {"degree": 3})]


def _weights(X, y):
    return AdaptiveCompositeKernel(KERNELS).fit(X, y).weights_


def test_alignment_weights_invariant_to_label_encoding():
    X, y = make_circles(400, noise=0.1, factor=0.4, random_state=0)
    ref = _weights(X, 2 * y - 1)
    for enc in (y, 1 - y, y * 5 + 3, np.where(y == 1, "pos", "neg")):
        np.testing.assert_allclose(_weights(X, enc), ref, atol=1e-12)


def test_multiclass_raises():
    X = np.random.default_rng(0).normal(size=(30, 2))
    y = np.repeat([0, 1, 2], 10)
    with pytest.raises(ValueError, match="binary"):
        AdaptiveCompositeKernel(KERNELS).fit(X, y)
