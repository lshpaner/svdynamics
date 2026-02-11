import numpy as np
from svdynamics import CompositeKernel


def test_composite_kernel_shapes_and_finiteness():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 5))
    Y = rng.normal(size=(12, 5))

    ck = CompositeKernel(
        kernels=[("rbf", {"gamma": 0.3}), "linear", ("poly", {"degree": 2, "coef0": 1.0})],
        weights=[0.5, 0.3, 0.2],
        normalize=True,
    )

    K = ck(X, Y)
    assert K.shape == (X.shape[0], Y.shape[0])
    assert np.all(np.isfinite(K))


def test_composite_kernel_weights_normalized():
    ck = CompositeKernel(kernels=["linear", "rbf"], weights=[2.0, 2.0], normalize=False)
    w = ck.normalized_weights
    assert np.isclose(w.sum(), 1.0)
    assert np.all(w > 0.0)
