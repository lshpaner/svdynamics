"""Tune C for composite kernels within a sklearn pipeline."""
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from svdynamics import AdaptiveCompositeKernel, CompositeKernel, SVDClassifier

X, y = make_classification(n_samples=350, n_features=8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42)
specs = ["linear", ("rbf", {"gamma": 0.2})]
pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("svm", SVDClassifier()),
])
search = GridSearchCV(
    pipeline,
    {
        "svm__C": [0.1, 1, 10],
        "svm__kernel": [
            CompositeKernel(specs, weights=[0.3, 0.7]),
            AdaptiveCompositeKernel(specs),
        ],
    },
    cv=5, scoring="accuracy", n_jobs=-1,
)
search.fit(X_train, y_train)
print("CV accuracy:", round(search.best_score_, 3))
print("Held-out accuracy:", round(search.score(X_test, y_test), 3))
print("Best C:", search.best_params_["svm__C"])
print("Best kernel type:", type(search.best_params_["svm__kernel"]).__name__)