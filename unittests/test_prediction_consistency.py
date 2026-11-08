import numpy as np
import pytest
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
def test_classifier_scores_do_not_depend_on_batch(data, kernel_cls):
    X, y = data
    model = SVDClassifier(kernel=kernel_cls(KERNELS)).fit(X, y)
    p = X[:5]
    in_batch = model.decision_function(np.vstack([p, X[5:200]]))[:5]
    alone = model.decision_function(p)
    one_by_one = np.array([model.decision_function(p[i : i + 1])[0] for i in range(5)])
    np.testing.assert_allclose(alone, in_batch, atol=1e-10)
    np.testing.assert_allclose(one_by_one, in_batch, atol=1e-10)


def test_regressor_predictions_do_not_depend_on_batch(data):
    X, y = data
    model = SVDRegressor(kernel=CompositeKernel(KERNELS)).fit(X, y.astype(float))
    p = X[:5]
    in_batch = model.predict(np.vstack([p, X[5:200]]))[:5]
    np.testing.assert_allclose(model.predict(p), in_batch, atol=1e-10)


def test_scales_come_from_training_data(data):
    X, y = data
    model = SVDClassifier(kernel=CompositeKernel(KERNELS)).fit(X, y)
    expected = CompositeKernel(KERNELS).fit_scales(X).scales_
    np.testing.assert_allclose(model.kernel_.scales_, expected)


def test_normalize_false_gives_unit_scales(data):
    X, _ = data
    k = CompositeKernel(KERNELS, normalize=False).fit_scales(X)
    np.testing.assert_array_equal(k.scales_, np.ones(3))
