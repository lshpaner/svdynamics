import numpy as np
from sklearn.datasets import make_classification, make_circles

from svdynamics import (
    AdaptiveCompositeKernel,
    CompositeKernel,
    SVDClassifier,
)


def test_svd_classifier_fit_predict():
    X, y = make_classification(
        n_samples=200,
        n_features=12,
        random_state=0,
    )

    ck = CompositeKernel(
        kernels=[
            ("rbf", {"gamma": 0.2}),
            "linear",
        ],
        weights=[0.7, 0.3],
        normalize=True,
    )

    clf = SVDClassifier(
        C=1.0,
        kernel=ck,
        probability=True,
        random_state=0,
    )
    clf.fit(X, y)

    pred = clf.predict(X[:10])
    proba = clf.predict_proba(X[:10])

    assert pred.shape == (10,)
    assert proba.shape == (10, 2)
    assert np.all(np.isfinite(proba))


def test_svd_classifier_adaptive_kernel_fit_predict():
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

    clf = SVDClassifier(
        kernel=kernel,
        random_state=42,
    )
    clf.fit(X, y)

    pred = clf.predict(X[:10])

    assert pred.shape == (10,)
    assert np.all(np.isfinite(pred))
    assert clf.kernel_.alignments_.shape == (3,)
    assert clf.kernel_.weights_.shape == (3,)
    assert np.all(np.isfinite(clf.kernel_.alignments_))
    assert np.all(np.isfinite(clf.kernel_.weights_))
    assert np.all(clf.kernel_.weights_ >= 0.0)
    assert np.isclose(clf.kernel_.weights_.sum(), 1.0)
