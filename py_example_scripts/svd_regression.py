from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR

from model_tuner.model_tuner_utils import Model

from svdynamics import CompositeKernel, SVDRegressor

# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------

data = load_diabetes(as_frame=True)
X = data["data"]
y = data["target"]

rstate = 42

# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

pipeline = [
    ("StandardScaler", StandardScaler()),
    ("Preprocessor", SimpleImputer()),
]

tuned_parameters = {
    "svm__C": [0.1, 1.0],
}

# ------------------------------------------------------------------
# Composite kernel
# ------------------------------------------------------------------

mixed_kernel = CompositeKernel(
    kernels=[
        ("rbf", {"gamma": 0.05}),
        ("linear", {}),
    ],
    weights=[0.7, 0.3],
    normalize=True,
)

estimator = SVDRegressor(
    kernel=mixed_kernel,
    C=1.0,
)

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

model = Model(
    name="SVR (Support Vector Dynamics)",
    estimator_name="svm",
    calibrate=False,
    model_type="regression",
    estimator=estimator,
    pipeline_steps=pipeline,
    kfold=True,
    stratify_y=False,
    grid=tuned_parameters,
    randomized_grid=False,
    n_iter=2,
    boost_early=False,
    scoring=["r2", "neg_root_mean_squared_error"],
    n_jobs=-1,
    random_state=rstate,
    imbalance_sampler=None,
)

# ------------------------------------------------------------------
# Grid search
# ------------------------------------------------------------------

model.grid_search_param_tuning(X, y)

# ------------------------------------------------------------------
# Fit
# ------------------------------------------------------------------

model.fit(X, y)

# ------------------------------------------------------------------
# Basic sanity checks
# ------------------------------------------------------------------

preds = model.predict(X)

assert preds.shape[0] == X.shape[0]
