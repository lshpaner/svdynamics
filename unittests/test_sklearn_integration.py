import copy

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from svdynamics import AdaptiveCompositeKernel, CompositeKernel, SVDClassifier

KERNELS = ["linear", ("rbf", {"gamma": 2.0}), ("poly", {"degree": 3})]


def test_gridsearch_with_adaptive_kernel():
    X, y = make_moons(300, noise=0.2, random_state=0)
    k = AdaptiveCompositeKernel(KERNELS)
    grid = GridSearchCV(
        make_pipeline(StandardScaler(), SVDClassifier(kernel=k)),
        {"svdclassifier__C": [0.1, 1.0, 10.0]},
        cv=3,
    ).fit(X, y)
    assert grid.best_score_ > 0.8
    assert not hasattr(k, "weights_")  # user's kernel untouched across all CV fits
    w = grid.best_estimator_[-1].kernel_.weights_
    assert np.isclose(w.sum(), 1.0)


def test_gridsearch_over_kernel_objects():
    X, y = make_moons(300, noise=0.2, random_state=0)
    grid = GridSearchCV(
        SVDClassifier(),
        {"kernel": [CompositeKernel(KERNELS), AdaptiveCompositeKernel(KERNELS), "rbf"]},
        cv=3,
    ).fit(X, y)
    assert grid.best_score_ > 0.8


def test_cross_val_score_runs():
    X, y = make_moons(300, noise=0.2, random_state=0)
    scores = cross_val_score(SVDClassifier(kernel=AdaptiveCompositeKernel(KERNELS)), X, y, cv=5)
    assert scores.shape == (5,)


def test_callable_kernel_survives_deepcopy():
    # Functions are copied by reference under deepcopy, so callable
    # kernels (including closures) work after the estimator copies the kernel.
    scale = {"g": 0.5}

    def my_rbf(A, B):
        d = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-scale["g"] * d)

    k = CompositeKernel(["linear", my_rbf])
    assert copy.deepcopy(k).parsed_kernels[1].fn is my_rbf
    X, y = make_moons(200, noise=0.2, random_state=0)
    assert SVDClassifier(kernel=k).fit(X, y).score(X, y) > 0.8
