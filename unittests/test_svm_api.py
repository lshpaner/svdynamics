"""Fitted-state checks, probability methods and SVC/SVR attribute parity."""
import numpy as np
import pytest
from sklearn.datasets import make_moons, make_regression
from sklearn.exceptions import NotFittedError

from svdynamics import CompositeKernel, SVDClassifier, SVDRegressor

KERNELS = ["linear", ("rbf", {"gamma": 1.0})]


@pytest.fixture
def clf_data():
    return make_moons(80, noise=0.2, random_state=0)


@pytest.fixture
def reg_data():
    return make_regression(80, n_features=3, noise=0.1, random_state=0)


@pytest.mark.parametrize("method", ["predict", "decision_function"])
def test_classifier_unfitted_raises(clf_data, method):
    X, _ = clf_data
    with pytest.raises(NotFittedError):
        getattr(SVDClassifier(), method)(X)


def test_regressor_unfitted_raises(reg_data):
    X, _ = reg_data
    with pytest.raises(NotFittedError):
        SVDRegressor().predict(X)


@pytest.mark.parametrize("attr", ["support_", "support_vectors_", "n_support_"])
def test_unfitted_attributes_raise(attr):
    for est in (SVDClassifier(), SVDRegressor()):
        with pytest.raises(NotFittedError):
            getattr(est, attr)


@pytest.mark.parametrize("method", ["predict_proba", "predict_log_proba"])
def test_probability_methods_hidden_without_probability(clf_data, method):
    X, y = clf_data
    clf = SVDClassifier().fit(X, y)
    assert not hasattr(clf, method)
    with pytest.raises(AttributeError, match="probability=True"):
        getattr(clf, method)


def test_probability_methods_with_probability(clf_data):
    X, y = clf_data
    clf = SVDClassifier(probability=True, random_state=0).fit(X, y)
    proba = clf.predict_proba(X[:5])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    np.testing.assert_allclose(clf.predict_log_proba(X[:5]), np.log(proba))


def test_classifier_attributes_match_svc(clf_data):
    X, y = clf_data
    clf = SVDClassifier(kernel=CompositeKernel(KERNELS)).fit(X, y)
    m = clf._model
    np.testing.assert_array_equal(clf.support_, m.support_)
    np.testing.assert_array_equal(clf.support_vectors_, m.support_vectors_)
    np.testing.assert_array_equal(clf.n_support_, m.n_support_)
    assert clf.n_support_.sum() == len(clf.support_)
    assert clf.n_features_in_ == X.shape[1]


def test_regressor_attributes_match_svr(reg_data):
    X, y = reg_data
    reg = SVDRegressor(kernel=CompositeKernel(KERNELS)).fit(X, y)
    m = reg._model
    np.testing.assert_array_equal(reg.support_, m.support_)
    np.testing.assert_array_equal(reg.support_vectors_, m.support_vectors_)
    np.testing.assert_array_equal(reg.n_support_, m.n_support_)
    assert reg.n_features_in_ == X.shape[1]


def test_n_iter_set_when_available(clf_data, reg_data):
    X, y = clf_data
    clf = SVDClassifier().fit(X, y)
    Xr, yr = reg_data
    reg = SVDRegressor().fit(Xr, yr)
    # SVC/SVR expose n_iter_ from scikit-learn 1.1 on; the wrappers mirror it
    for est in (clf, reg):
        assert hasattr(est, "n_iter_") == hasattr(est._model, "n_iter_")
