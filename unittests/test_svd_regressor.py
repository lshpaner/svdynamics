import numpy as np
from sklearn.datasets import make_regression
from svdynamics import CompositeKernel, SVDRegressor


def test_svd_regressor_fit_predict():
    X, y = make_regression(n_samples=180, n_features=10, noise=0.2, random_state=0)

    ck = CompositeKernel(
        kernels=[("rbf", {"gamma": 0.15}), ("poly", {"degree": 2, "coef0": 1.0})],
        weights=[0.6, 0.4],
        normalize=True,
    )

    reg = SVDRegressor(C=2.0, epsilon=0.2, kernel=ck)
    reg.fit(X, y)

    pred = reg.predict(X[:12])
    assert pred.shape == (12,)
    assert np.all(np.isfinite(pred))
