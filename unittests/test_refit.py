import numpy as np
from sklearn.datasets import make_circles, make_moons

from svdynamics import AdaptiveCompositeKernel, CompositeKernel

KERNELS = ["linear", ("rbf", {"gamma": 2.0}), ("poly", {"degree": 3})]


def _two_datasets():
    Xa, ya = make_circles(300, noise=0.1, factor=0.4, random_state=0)
    Xb, yb = make_moons(250, noise=0.2, random_state=1)
    return (Xa, ya), (Xb * 2.5, yb)


def test_fit_scales_refit_matches_fresh_fit():
    (Xa, _), (Xb, _) = _two_datasets()
    k = CompositeKernel(KERNELS)
    k.fit_scales(Xa)
    first = k.scales_.copy()
    k.fit_scales(Xb)
    fresh = CompositeKernel(KERNELS).fit_scales(Xb).scales_
    np.testing.assert_allclose(k.scales_, fresh)
    assert not np.allclose(first, k.scales_)


def test_adaptive_refit_matches_fresh_fit():
    (Xa, ya), (Xb, yb) = _two_datasets()
    k = AdaptiveCompositeKernel(KERNELS)
    k.fit(Xa, ya)
    first_w = k.weights_.copy()
    k.fit(Xb, yb)
    fresh = AdaptiveCompositeKernel(KERNELS).fit(Xb, yb)
    np.testing.assert_allclose(k.weights_, fresh.weights_)
    np.testing.assert_allclose(k.alignments_, fresh.alignments_)
    np.testing.assert_allclose(k.scales_, fresh.scales_)
    np.testing.assert_allclose(k.normalized_weights, fresh.normalized_weights)
    assert not np.allclose(first_w, k.weights_)


def test_refit_kernel_values_match_fresh_fit():
    (Xa, ya), (Xb, yb) = _two_datasets()
    k = AdaptiveCompositeKernel(KERNELS).fit(Xa, ya).fit(Xb, yb)
    fresh = AdaptiveCompositeKernel(KERNELS).fit(Xb, yb)
    np.testing.assert_allclose(k(Xb[:10], Xb), fresh(Xb[:10], Xb))
