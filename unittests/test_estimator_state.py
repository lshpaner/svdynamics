import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_circles

from svdynamics import (
    AdaptiveCompositeKernel,
    CompositeKernel,
    SVDClassifier,
    SVDRegressor,
)

KERNELS = ["linear", ("rbf", {"gamma": 2.0}), ("poly", {"degree": 3})]


@pytest.fixture
def data():
    X, y = make_circles(400, noise=0.1, factor=0.4, random_state=0)
    return X, y


@pytest.mark.parametrize("kernel_cls", [CompositeKernel, AdaptiveCompositeKernel])
def test_fit_does_not_modify_kernel_param(data, kernel_cls):
    X, y = data
    k = kernel_cls(KERNELS)
    model = SVDClassifier(kernel=k).fit(X, y)
    assert model.kernel is k
    assert not hasattr(k, "scales_")
    assert not hasattr(k, "weights_")
    assert model.kernel_ is not k
    assert hasattr(model.kernel_, "scales_")


def test_adaptive_weights_on_fitted_kernel(data):
    X, y = data
    model = SVDClassifier(kernel=AdaptiveCompositeKernel(KERNELS)).fit(X, y)
    w = model.kernel_.weights_
    assert w.shape == (3,)
    assert np.isclose(w.sum(), 1.0)


def test_clone_refit_matches(data):
    X, y = data
    a = SVDClassifier(kernel=AdaptiveCompositeKernel(KERNELS)).fit(X, y)
    b = clone(a).fit(X, y)
    np.testing.assert_allclose(a.decision_function(X), b.decision_function(X))
    np.testing.assert_allclose(a.kernel_.weights_, b.kernel_.weights_)


def test_same_kernel_two_datasets_independent(data):
    X, y = data
    k = AdaptiveCompositeKernel(KERNELS)
    m1 = SVDClassifier(kernel=k).fit(X, y)
    m2 = SVDClassifier(kernel=k).fit(X * 3.0, y)
    assert not np.allclose(m1.kernel_.scales_, m2.kernel_.scales_)
    # m1 is unaffected by fitting m2
    np.testing.assert_allclose(
        m1.decision_function(X[:5]),
        SVDClassifier(kernel=AdaptiveCompositeKernel(KERNELS)).fit(X, y).decision_function(X[:5]),
    )


def test_regressor_rejects_adaptive(data):
    X, y = data
    with pytest.raises(TypeError, match="not supported for regression"):
        SVDRegressor(kernel=AdaptiveCompositeKernel(KERNELS)).fit(X, y.astype(float))


def test_unfitted_cross_call_warns(data):
    X, _ = data
    with pytest.warns(UserWarning, match="fit_scales"):
        CompositeKernel(KERNELS)(X[:5], X)


def test_unfitted_gram_and_fitted_calls_do_not_warn(data):
    X, _ = data
    k = CompositeKernel(KERNELS)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        k(X, X)
        k.fit_scales(X)(X[:5], X)
