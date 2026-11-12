
"""Nested CV benchmark of composite SVM kernel-weight selection on MAGIC."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from svdynamics import (
    AdaptiveCompositeKernel,
    CompositeKernel,
    SVDClassifier,
)


DATA_PATH = (
    Path(__file__).resolve().parent / "data" / "magic_sample.csv"
)

BASE_FEATURES = ("fLength", "fWidth")

GAMMA = 0.5
OUTER_SPLITS = 5
OUTER_REPEATS = 3
INNER_SPLITS = 3
SEED = 42

WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
C_VALUES = (0.1, 1.0, 10.0)


def group_kernel(indices):
    """Construct an RBF kernel using selected feature columns."""

    indices = np.asarray(indices, dtype=int)

    def kernel(x, y):
        return rbf_kernel(
            np.asarray(x)[:, indices],
            np.asarray(y)[:, indices],
            gamma=GAMMA,
        )

    return kernel


def make_model(base_indices, increment_indices, c, weight=None):
    """Create an adaptive or manually weighted composite SVM."""

    kernels = [
        group_kernel(base_indices),
        group_kernel(increment_indices),
    ]

    if weight is None:
        kernel = AdaptiveCompositeKernel(
            kernels=kernels,
            normalize=False,
        )
    else:
        kernel = CompositeKernel(
            kernels=kernels,
            weights=[weight, 1.0 - weight],
            normalize=False,
        )

    return SVDClassifier(
        C=c,
        kernel=kernel,
    )


def metrics(model, x_train, y_train, x_test, y_test):
    """Fit a model and calculate accuracy and ROC-AUC."""

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
            "Missing MAGIC sample; run make research_magic_data"
        )

    frame = pd.read_csv(DATA_PATH)

    if "target" not in frame.columns:
        raise ValueError("Missing target column")

    x_frame = frame.drop(columns="target")
    y = frame["target"].to_numpy()

    if set(np.unique(y)) != {-1, 1}:
        raise ValueError("Expected labels -1 and +1")

    base = [
        x_frame.columns.get_loc(name)
        for name in BASE_FEATURES
    ]

    increment = [
        i for i in range(x_frame.shape[1])
        if i not in base
    ]

    x = x_frame.to_numpy(dtype=float)

    outer = RepeatedStratifiedKFold(
        n_splits=OUTER_SPLITS,
        n_repeats=OUTER_REPEATS,
        random_state=SEED,
    )

    rows = []

    for fold, (train_idx, test_idx) in enumerate(
        outer.split(x, y), start=1
    ):
        x_outer_train = x[train_idx]
        x_outer_test = x[test_idx]

        y_outer_train = y[train_idx]
        y_outer_test = y[test_idx]

        inner = StratifiedKFold(
            n_splits=INNER_SPLITS,
            shuffle=True,
            random_state=SEED + fold,
        )

        # Fit preprocessing separately within each inner fold.
        inner_data = []

        for in_train, in_valid in inner.split(
            x_outer_train, y_outer_train
        ):
            scaler = StandardScaler().fit(
                x_outer_train[in_train]
            )

            inner_data.append((
                scaler.transform(x_outer_train[in_train]),
                y_outer_train[in_train],
                scaler.transform(x_outer_train[in_valid]),
                y_outer_train[in_valid],
            ))

        candidates = []

        for c in C_VALUES:
            for weight in WEIGHTS:
                aucs = []

                for x_tr, y_tr, x_va, y_va in inner_data:
                    model = make_model(
                        base,
                        increment,
                        c,
                        weight,
                    )

                    _, auc = metrics(
                        model,
                        x_tr,
                        y_tr,
                        x_va,
                        y_va,
                    )

                    aucs.append(auc)

                candidates.append((
                    float(np.mean(aucs)),
                    c,
                    weight,
                ))

        # Deterministic tie-breaking:
        # lower C, then weight closest to 0.5.
        best_auc, best_c, best_weight = sorted(
            candidates,
            key=lambda t: (
                -t[0],
                t[1],
                abs(t[2] - 0.5),
                t[2],
            ),
        )[0]

        # Refit preprocessing on the full outer training fold.
        outer_scaler = StandardScaler().fit(x_outer_train)

        x_tr = outer_scaler.transform(x_outer_train)
        x_te = outer_scaler.transform(x_outer_test)

        equal = make_model(
            base,
            increment,
            1.0,
            0.5,
        )

        adaptive = make_model(
            base,
            increment,
            1.0,
        )

        tuned = make_model(
            base,
            increment,
            best_c,
            best_weight,
        )

        equal_acc, equal_auc = metrics(
            equal,
            x_tr,
            y_outer_train,
            x_te,
            y_outer_test,
        )

        adaptive_acc, adaptive_auc = metrics(
            adaptive,
            x_tr,
            y_outer_train,
            x_te,
            y_outer_test,
        )

        tuned_acc, tuned_auc = metrics(
            tuned,
            x_tr,
            y_outer_train,
            x_te,
            y_outer_test,
        )

        # SVDClassifier stores the fitted kernel in kernel_.
        learned = adaptive.kernel_.weights_

        rows.append({
            "fold": fold,
            "equal_accuracy": equal_acc,
            "adaptive_accuracy": adaptive_acc,
            "tuned_accuracy": tuned_acc,
            "equal_auc": equal_auc,
            "adaptive_auc": adaptive_auc,
            "tuned_auc": tuned_auc,
            "delta_equal": tuned_auc - equal_auc,
            "delta_adaptive": tuned_auc - adaptive_auc,
            "chosen_c": best_c,
            "chosen_base_weight": best_weight,
            "alignment_base_weight": float(learned[0]),
            "inner_auc": best_auc,
        })

        print(
            "Fold {:02d}/15 | Equal {:.4f} | "
            "Adaptive {:.4f} | Tuned {:.4f} | "
            "C={} | Base weight={:.2f}".format(
                fold,
                equal_auc,
                adaptive_auc,
                tuned_auc,
                best_c,
                best_weight,
            ),
            flush=True,
        )

    result = pd.DataFrame(rows)

    print("\n--- MAGIC: Nested kernel-weight optimization ---")
    print("Outer: 5 folds x 3 repeats; inner: 3 stratified folds")

    print(
        "RBF gamma={}; weights={}; C={}".format(
            GAMMA,
            WEIGHTS,
            C_VALUES,
        )
    )

    print(
        result.drop(columns="fold")
        .agg(["mean", "std"])
        .round(5)
        .to_string()
    )

    for col in ("delta_equal", "delta_adaptive"):
        d = result[col].to_numpy()

        wins = int((d > 1e-12).sum())
        ties = int((np.abs(d) <= 1e-12).sum())
        losses = int((d < -1e-12).sum())

        print(
            "{} wins/ties/losses: {}".format(
                col,
                (wins, ties, losses),
            )
        )

    print("Chosen (C, base weight) frequencies:")

    print(
        result.groupby([
            "chosen_c",
            "chosen_base_weight",
        ]).size().to_string()
    )

    print(
        "Note: repeated outer folds overlap; "
        "comparisons are descriptive."
    )


if __name__ == "__main__":
    main()
