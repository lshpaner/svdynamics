import numpy as np

from sklearn.datasets import make_classification
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
)

from svdynamics import CompositeKernel, SVDClassifier


X, y = make_classification(
    n_samples=300,
    n_features=10,
    random_state=0,
)

kernel = CompositeKernel(
    kernels=[
        ("rbf", {"gamma": 0.2}),
        ("linear", {}),
        ("poly", {"degree": 2, "coef0": 1.0}),
    ],
    weights=[0.6, 0.3, 0.1],
    normalize=True,
)

clf = SVDClassifier(
    C=1.0,
    kernel=kernel,
    probability=True,
    random_state=0,
)

clf.fit(X, y)

# --- predictions on the full dataset (for metrics) ---
y_pred = clf.predict(X)
y_proba = clf.predict_proba(X)[:, 1]

# --- metrics ---
auc = roc_auc_score(y, y_proba)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
acc = accuracy_score(y, y_pred)

print("\n=== Classification metrics (training set) ===")
print(f"Accuracy : {acc:.4f}")
print(f"ROC AUC  : {auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 score : {f1:.4f}")

print("\nClassification report:")
print(classification_report(y, y_pred))

# --- quick sanity check on a few rows ---
proba_5 = clf.predict_proba(X[:5])
pred_5 = clf.predict(X[:5])

print("\nFirst 5 predicted probabilities:")
print(proba_5)

print("\nFirst 5 predicted labels:")
print(pred_5)
