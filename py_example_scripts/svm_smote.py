from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

from model_tuner.model_tuner_utils import Model
import model_tuner
from imblearn.over_sampling import SMOTE

from svdynamics import CompositeKernel, SVDClassifier


print()
print(f"Model Tuner version: {model_tuner.__version__}")
print(f"Model Tuner authors: {model_tuner.__author__}")
print()

# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------

bc = load_breast_cancer(as_frame=True)["frame"]
bc_cols = [c for c in bc.columns if c != "target"]
X = bc[bc_cols]
y = bc["target"]

rstate = 42

print("Data shape:", X.shape)

# ------------------------------------------------------------------
# Shared config
# ------------------------------------------------------------------

pipeline = [
    ("StandardScalar", StandardScaler()),
    ("Preprocessor", SimpleImputer()),
]

tuned_parameters = {
    "svm__C": [1, 10],
}

kfold = True
calibrate = True


def run_model(name, estimator):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    model = Model(
        name=name,
        estimator_name="svm",
        calibrate=calibrate,
        model_type="classification",
        estimator=estimator,
        pipeline_steps=pipeline,
        kfold=kfold,
        stratify_y=True,
        grid=tuned_parameters,
        randomized_grid=False,
        n_iter=4,
        boost_early=False,
        scoring=["roc_auc"],
        n_jobs=-2,
        random_state=rstate,
        imbalance_sampler=SMOTE(random_state=rstate),
    )

    model.grid_search_param_tuning(X, y, f1_beta_tune=True)
    model.fit(X, y)

    if model.calibrate:
        model.calibrateModel(
            X,
            y,
            score="roc_auc",
            f1_beta_tune=True
        )

    print("Validation Metrics")
    model.return_metrics(X, y, print_threshold=True, model_metrics=True)


# ------------------------------------------------------------------
# 1) Mixed kernel (Support Vector Dynamics)
# ------------------------------------------------------------------

mixed_kernel = CompositeKernel(
    kernels=[
        ("rbf", {"gamma": 0.05}),
        ("linear", {}),
    ],
    weights=[0.7, 0.3],
    normalize=True,
)

svd_estimator = SVDClassifier(
    kernel=mixed_kernel,
    C=1.0,
    probability=True,
    random_state=rstate,
)

run_model(
    name="SVM (mixed kernel: rbf + linear => Support Vector Dynamics)",
    estimator=svd_estimator,
)

# ------------------------------------------------------------------
# 2) Plain RBF SVM
# ------------------------------------------------------------------

rbf_estimator = SVC(
    kernel="rbf",
    C=1.0,
    probability=True,
    class_weight=None,
    random_state=rstate,
)

run_model(
    name="SVM (plain rbf kernel, sklearn)",
    estimator=rbf_estimator,
)

# ------------------------------------------------------------------
# 3) Plain linear SVM
# ------------------------------------------------------------------

linear_estimator = SVC(
    kernel="linear",
    C=1.0,
    probability=True,
    class_weight=None,
    random_state=rstate,
)

run_model(
    name="SVM (plain linear kernel, sklearn)",
    estimator=linear_estimator,
)
