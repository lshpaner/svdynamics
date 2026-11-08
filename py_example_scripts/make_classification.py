"""Held-out classification comparison for standard and composite SVMs."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from svdynamics import AdaptiveCompositeKernel, CompositeKernel, SVDClassifier

X, y = make_classification(n_samples=500, n_features=10, n_informative=6,
                           n_redundant=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
specs = [("rbf", {"gamma": 0.1}), "linear", ("poly", {"degree": 2, "coef0": 1})]
models = {
    "RBF SVC": SVC(kernel="rbf", C=1, probability=True, random_state=42),
    "Linear SVC": SVC(kernel="linear", C=1, probability=True, random_state=42),
    "Fixed composite": SVDClassifier(
        kernel=CompositeKernel(specs, weights=[0.6, 0.3, 0.1]),
        probability=True, random_state=42),
    "Adaptive composite": SVDClassifier(
        kernel=AdaptiveCompositeKernel(specs),
        probability=True, random_state=42),
}
for name, estimator in models.items():
    pipe = make_pipeline(StandardScaler(), estimator)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    fitted = pipe.steps[-1][1]
    proba = pipe.predict_proba(X_test)[:, list(fitted.classes_).index(1)]
    print(f"{name:20} accuracy={accuracy_score(y_test, pred):.3f} "
          f"balanced_acc={balanced_accuracy_score(y_test, pred):.3f} "
          f"F1={f1_score(y_test, pred):.3f} "
          f"AUC={roc_auc_score(y_test, proba):.3f} "
          f"AP={average_precision_score(y_test, proba):.3f} "
          f"Brier={brier_score_loss(y_test, proba):.3f}")
    if name == "Adaptive composite":
        print("  Learned weights:", np.round(fitted.kernel_.weights_, 3))
        print("  Alignments:", np.round(fitted.kernel_.alignments_, 3))