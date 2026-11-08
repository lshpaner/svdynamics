"""Imbalanced classification with leakage-safe SMOTE inside CV folds.

Optional dependency: pip install imbalanced-learn
"""
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from svdynamics import AdaptiveCompositeKernel, CompositeKernel, SVDClassifier

X, y = make_classification(n_samples=900, n_features=12, n_informative=7,
                           n_redundant=2, weights=[0.9, 0.1],
                           flip_y=0.01, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
specs = [("rbf", {"gamma": 0.1}), "linear"]
models = {
    "RBF": SVC(kernel="rbf", probability=True, random_state=42),
    "Fixed": SVDClassifier(kernel=CompositeKernel(specs, weights=[0.7, 0.3]),
                           probability=True, random_state=42),
    "Adaptive": SVDClassifier(kernel=AdaptiveCompositeKernel(specs),
                              probability=True, random_state=42),
}
for name, estimator in models.items():
    # SMOTE is fitted only on training partitions, including inside each CV fold.
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("smote", SMOTE(random_state=42, k_neighbors=3)),
        ("svm", estimator),
    ])
    search = GridSearchCV(pipeline, {"svm__C": [0.1, 1, 10]},
                          scoring="average_precision", cv=5, n_jobs=-1)
    search.fit(X_train, y_train)
    scores = search.predict_proba(X_test)[:, 1]
    preds = search.predict(X_test)
    print(f"{name:9} AP={average_precision_score(y_test, scores):.3f} "
          f"AUC={roc_auc_score(y_test, scores):.3f} "
          f"balanced_acc={balanced_accuracy_score(y_test, preds):.3f} "
          f"best_C={search.best_params_['svm__C']}")
# Final test set is never used for fitting, tuning, or threshold selection.