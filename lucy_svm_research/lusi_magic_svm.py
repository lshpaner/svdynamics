
"""Evaluate the actual svdynamics adaptive SVM on MAGIC.

Compare:
1. RBF SVM using two baseline features
2. RBF SVM using all ten features
3. Equal-weight composite SVM
4. Alignment-weighted adaptive composite SVM

All methods use identical repeated stratified CV folds.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from svdynamics import (
    AdaptiveCompositeKernel,
    CompositeKernel,
    SVDClassifier,
)


DATA_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "magic_sample.csv"
)

BASE_FEATURES = ["fLength", "fWidth"]

N_SPLITS = 5
N_REPEATS = 3
RANDOM_STATE = 42

C = 1.0
GAMMA = 0.5


def make_group_kernel(indices):
    """Return an RBF kernel operating on selected columns."""

    indices = np.asarray(indices, dtype=int)

    def kernel(x, y):
        return rbf_kernel(
            np.asarray(x)[:, indices],
            np.asarray(y)[:, indices],
            gamma=GAMMA,
        )

    return kernel


def evaluate(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    scores = model.decision_function(x_test)

    return (
        accuracy_score(y_test, predictions),
        roc_auc_score(y_test, scores),
    )


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            "Run make research_magic_data first"
        )

    frame = pd.read_csv(DATA_PATH)

    if "target" not in frame.columns:
        raise ValueError("Missing target column")

    x = frame.drop(columns="target")
    y = frame["target"].to_numpy()

    if set(np.unique(y)) != {-1, 1}:
        raise ValueError("Expected -1/+1 labels")

    base_indices = [
        x.columns.get_loc(name)
        for name in BASE_FEATURES
    ]

    increment_indices = [
        i for i in range(x.shape[1])
        if i not in base_indices
    ]

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    results = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(x, y), start=1
    ):
        scaler = StandardScaler()

        x_train = scaler.fit_transform(
            x.iloc[train_idx]
        )
        x_test = scaler.transform(
            x.iloc[test_idx]
        )

        y_train = y[train_idx]
        y_test = y[test_idx]

        base_kernel = make_group_kernel(base_indices)
        increment_kernel = make_group_kernel(
            increment_indices
        )

        base_svm = SVC(
            C=C,
            kernel="rbf",
            gamma=GAMMA,
        )

        full_svm = SVC(
            C=C,
            kernel="rbf",
            gamma=GAMMA,
        )

        equal_kernel = CompositeKernel(
            kernels=[base_kernel, increment_kernel],
            weights=[0.5, 0.5],
            normalize=False,
        )

        adaptive_kernel = AdaptiveCompositeKernel(
            kernels=[
                make_group_kernel(base_indices),
                make_group_kernel(increment_indices),
            ],
            normalize=False,
        )

        equal_svm = SVDClassifier(
            C=C,
            kernel=equal_kernel,
        )

        adaptive_svm = SVDClassifier(
            C=C,
            kernel=adaptive_kernel,
        )

        base_acc, base_auc = evaluate(
            base_svm,
            x_train[:, base_indices],
            x_test[:, base_indices],
            y_train,
            y_test,
        )

        full_acc, full_auc = evaluate(
            full_svm,
            x_train,
            x_test,
            y_train,
            y_test,
        )

        equal_acc, equal_auc = evaluate(
            equal_svm,
            x_train,
            x_test,
            y_train,
            y_test,
        )

        adaptive_acc, adaptive_auc = evaluate(
            adaptive_svm,
            x_train,
            x_test,
            y_train,
            y_test,
        )

        weights = adaptive_svm.kernel_.weights_

        results.append({
            "fold": fold,
            "base_accuracy": base_acc,
            "base_auc": base_auc,
            "full_accuracy": full_acc,
            "full_auc": full_auc,
            "equal_accuracy": equal_acc,
            "equal_auc": equal_auc,
            "adaptive_accuracy": adaptive_acc,
            "adaptive_auc": adaptive_auc,
            "delta_auc_equal": adaptive_auc - equal_auc,
            "delta_auc_full": adaptive_auc - full_auc,
            "base_weight": weights[0],
            "increment_weight": weights[1],
        })

        print(
            "Fold {:02d}/{:02d} | "
            "Full AUC={:.4f} | "
            "Equal AUC={:.4f} | "
            "Adaptive AUC={:.4f} | "
            "Weights=({:.3f}, {:.3f})".format(
                fold,
                N_SPLITS * N_REPEATS,
                full_auc,
                equal_auc,
                adaptive_auc,
                weights[0],
                weights[1],
            ),
            flush=True,
        )

    output = pd.DataFrame(results)

    print("\n--- MAGIC: Actual Adaptive SVM Benchmark ---")
    print(
        "Outer: {} folds x {} repeats".format(
            N_SPLITS, N_REPEATS
        )
    )
    print("SVM C={}; RBF gamma={}".format(C, GAMMA))
    print("Scaling fitted separately on each training fold")

    columns = [
        "base_accuracy",
        "full_accuracy",
        "equal_accuracy",
        "adaptive_accuracy",
        "base_auc",
        "full_auc",
        "equal_auc",
        "adaptive_auc",
        "delta_auc_equal",
        "delta_auc_full",
        "base_weight",
        "increment_weight",
    ]

    print(
        output[columns]
        .agg(["mean", "std"])
        .round(5)
        .to_string()
    )

    for column in ["delta_auc_equal", "delta_auc_full"]:
        values = output[column]

        wins = int((values > 1e-10).sum())
        ties = int((values.abs() <= 1e-10).sum())
        losses = int((values < -1e-10).sum())

        print(
            "{} wins/ties/losses: {}".format(
                column,
                (wins, ties, losses),
            )
        )

    print(
        "Note: repeated folds overlap; results are descriptive."
    )


if __name__ == "__main__":
    main()
