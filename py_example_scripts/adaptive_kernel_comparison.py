"""Repeated-CV benchmark. All scaling and adaptive fitting occur within folds."""
import numpy as np
from sklearn.datasets import make_circles, make_moons, make_classification
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from svdynamics import AdaptiveCompositeKernel, CompositeKernel, SVDClassifier

datasets = {
    "moons": make_moons(n_samples=300, noise=0.22, random_state=42),
    "circles": make_circles(n_samples=300, noise=0.12, factor=0.45, random_state=42),
    "classification": make_classification(
        n_samples=300, n_features=8, n_informative=5, n_redundant=1,
        random_state=42),
}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
specs = [("rbf", {"gamma": 0.5}), "linear", ("poly", {"degree": 2, "coef0": 1})]
models = {
    "RBF": SVC(kernel="rbf", gamma=0.5),
    "Linear": SVC(kernel="linear"),
    "Polynomial": SVC(kernel="poly", degree=2, coef0=1),
    "Fixed": SVDClassifier(kernel=CompositeKernel(specs, weights=[0.5, 0.25, 0.25])),
    "Adaptive": SVDClassifier(kernel=AdaptiveCompositeKernel(specs)),
}
for dataset, (X, y) in datasets.items():
    print(f"\n{dataset}")
    for name, model in models.items():
        scores = cross_validate(make_pipeline(StandardScaler(), model), X, y,
                                cv=cv, scoring={"acc": "accuracy", "balanced": "balanced_accuracy"})
        print(f"  {name:12} accuracy={scores['test_acc'].mean():.3f}"
              f" +/- {scores['test_acc'].std():.3f} "
              f"balanced_acc={scores['test_balanced'].mean():.3f}")