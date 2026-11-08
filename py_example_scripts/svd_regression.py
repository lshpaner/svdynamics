"""Diabetes regression with a held-out test set and a CV-tuned composite."""
from sklearn.datasets import load_diabetes
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from svdynamics import CompositeKernel, SVDRegressor

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
estimators = {
    "RBF SVR": SVR(kernel="rbf"),
    "Linear SVR": SVR(kernel="linear"),
    "Composite SVR": SVDRegressor(
        kernel=CompositeKernel(
            [("rbf", {"gamma": 0.05}), "linear"],
            weights=[0.7, 0.3], normalize=True)),
}
for name, estimator in estimators.items():
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svm", estimator),
    ])
    search = GridSearchCV(pipeline, {"svm__C": [0.1, 1, 10, 100]},
                          scoring="neg_root_mean_squared_error", cv=5, n_jobs=-1)
    search.fit(X_train, y_train)
    pred = search.predict(X_test)
    print(f"{name:16} R2={r2_score(y_test, pred):.3f} "
          f"RMSE={mean_squared_error(y_test, pred) ** 0.5:.3f} "
          f"MAE={mean_absolute_error(y_test, pred):.3f} "
          f"best_C={search.best_params_['svm__C']}")