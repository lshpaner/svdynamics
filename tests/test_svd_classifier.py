import numpy as np
from sklearn.datasets import make_classification
from svdynamics import CompositeKernel, SVDClassifier


def test_svd_classifier_fit_predict():
    X, y = make_classification(n_samples=200, n_features=12, random_state=0)

    ck = CompositeKernel(
        kernels=[("rbf", {"gamma": 0.2}), "linear"],
        weights=[0.7, 0.3],
        normalize=True,
    )

    clf = SVDClassifier(C=1.0, kernel=ck, probability=True, random_state=0)
    clf.fit(X, y)

    pred = clf.predict(X[:10])
    proba = clf.predict_proba(X[:10])

    assert pred.shape == (10,)
    assert proba.shape == (10, 2)
    assert np.all(np.isfinite(proba))
