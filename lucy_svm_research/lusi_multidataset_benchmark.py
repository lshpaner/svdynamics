
"""Multi-dataset benchmark for composite SVM kernel weighting.

Compare:
1. Equal-weight composite SVM
2. Alignment-weighted adaptive composite SVM
3. Nested-CV optimized composite SVM

All preprocessing and model selection occur within training folds.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_wine,
    make_circles,
    make_moons,
)
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler

from svdynamics import (
    AdaptiveCompositeKernel,
    CompositeKernel,
    SVDClassifier,
)


GAMMA = 0.5
OUTER_SPLITS = 5
OUTER_REPEATS = 2
INNER_SPLITS = 3
SEED = 42

C_VALUES = (0.1, 1.0, 10.0)
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)

OUTPUT_DIR = (
    Path(__file__).resolve().parent / "results"
)


def load_datasets():
    """Load reproducible binary classification datasets."""

    cancer = load_breast_cancer()
    wine = load_wine()
    digits = load_digits()

    wine_mask = wine.target != 2
    digits_mask = digits.target <= 1

    moons_x, moons_y = make_moons(
        n_samples=600,
        noise=0.25,
        random_state=SEED,
    )

    circles_x, circles_y = make_circles(
        n_samples=600,
        noise=0.12,
        factor=0.45,
        random_state=SEED,
    )

    return {
        "Breast Cancer": (
            cancer.data,
            cancer.target,
        ),
        "Wine 0 vs 1": (
            wine.data[wine_mask],
            wine.target[wine_mask],
        ),
        "Digits 0 vs 1": (
            digits.data[digits_mask],
            digits.target[digits_mask],
        ),
        "Moons": (
            moons_x,
            moons_y,
        ),
        "Circles": (
            circles_x,
            circles_y,
        ),
    }


def make_group_kernel(indices):
    """Build an RBF kernel using a subset of columns."""

    indices = np.asarray(indices, dtype=int)

    def kernel(x, y):
        x = np.asarray(x)
        y = np.asarray(y)

        return rbf_kernel(
            x[:, indices],
            y[:, indices],
            gamma=GAMMA,
        )

    return kernel


def make_model(base, increment, c, weight=None):
    """Construct a weighted or adaptive SVDClassifier."""

    kernels = [
        make_group_kernel(base),
        make_group_kernel(increment),
    ]

    if weight is None:
        kernel = AdaptiveCompositeKernel(
            kernels=kernels,
            normalize=False,
        )
    else:
        kernel = CompositeKernel(
            kernels=kernels,
            weights=[
                float(weight),
                float(1.0 - weight),
            ],
            normalize=False,
        )

    return SVDClassifier(
        C=float(c),
        kernel=kernel,
    )


def evaluate(model, x_train, y_train, x_test, y_test):
    """Fit model and return accuracy and ROC-AUC."""

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    scores = model.decision_function(x_test)

    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, scores)

    return float(accuracy), float(auc)


def optimize_weights(x, y, base, increment, fold):
    """Select C and base weight using inner cross-validation."""

    inner = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=SEED + fold,
    )

    prepared = []

    for train_idx, valid_idx in inner.split(x, y):
        scaler = StandardScaler()

        x_train = scaler.fit_transform(x[train_idx])
        x_valid = scaler.transform(x[valid_idx])

        prepared.append((
            x_train,
            y[train_idx],
            x_valid,
            y[valid_idx],
        ))

    candidates = []

    for c in C_VALUES:
        for weight in WEIGHTS:
            aucs = []

            for x_train, y_train, x_valid, y_valid in prepared:
                model = make_model(
                    base,
                    increment,
                    c,
                    weight,
                )

                _, auc = evaluate(
                    model,
                    x_train,
                    y_train,
                    x_valid,
                    y_valid,
                )

                aucs.append(auc)

            candidates.append({
                "c": c,
                "weight": weight,
                "auc": float(np.mean(aucs)),
            })

    # Prefer lower C, then weight closest to 0.5
    # when inner-CV scores are exactly tied.
    best = sorted(
        candidates,
        key=lambda row: (
            -row["auc"],
            row["c"],
            abs(row["weight"] - 0.5),
            row["weight"],
        ),
    )[0]

    return best


def benchmark_dataset(name, x, y):
    """Run nested repeated CV on one dataset."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y)

    if len(np.unique(y)) != 2:
        raise ValueError(
            "{} is not binary".format(name)
        )

    n_features = x.shape[1]
    midpoint = n_features // 2

    base = np.arange(midpoint)
    increment = np.arange(midpoint, n_features)

    outer = RepeatedStratifiedKFold(
        n_splits=OUTER_SPLITS,
        n_repeats=OUTER_REPEATS,
        random_state=SEED,
    )

    rows = []

    print(
        "\nDataset: {} | N={} | Features={}".format(
            name,
            len(y),
            n_features,
        ),
        flush=True,
    )

    for fold, (train_idx, test_idx) in enumerate(
        outer.split(x, y),
        start=1,
    ):
        x_outer_train = x[train_idx]
        x_outer_test = x[test_idx]

        y_outer_train = y[train_idx]
        y_outer_test = y[test_idx]

        best = optimize_weights(
            x_outer_train,
            y_outer_train,
            base,
            increment,
            fold,
        )

        scaler = StandardScaler()

        x_train = scaler.fit_transform(x_outer_train)
        x_test = scaler.transform(x_outer_test)

        equal = make_model(
            base,
            increment,
            c=1.0,
            weight=0.5,
        )

        adaptive = make_model(
            base,
            increment,
            c=1.0,
        )

        tuned = make_model(
            base,
            increment,
            c=best["c"],
            weight=best["weight"],
        )

        equal_acc, equal_auc = evaluate(
            equal,
            x_train,
            y_outer_train,
            x_test,
            y_outer_test,
        )

        adaptive_acc, adaptive_auc = evaluate(
            adaptive,
            x_train,
            y_outer_train,
            x_test,
            y_outer_test,
        )

        tuned_acc, tuned_auc = evaluate(
            tuned,
            x_train,
            y_outer_train,
            x_test,
            y_outer_test,
        )

        alignment_weights = adaptive.kernel_.weights_

        rows.append({
            "dataset": name,
            "fold": fold,
            "n_samples": len(y),
            "n_features": n_features,
            "equal_accuracy": equal_acc,
            "adaptive_accuracy": adaptive_acc,
            "tuned_accuracy": tuned_acc,
            "equal_auc": equal_auc,
            "adaptive_auc": adaptive_auc,
            "tuned_auc": tuned_auc,
            "delta_equal": tuned_auc - equal_auc,
            "delta_adaptive": tuned_auc - adaptive_auc,
            "chosen_c": best["c"],
            "chosen_base_weight": best["weight"],
            "alignment_base_weight": float(
                alignment_weights[0]
            ),
            "inner_auc": best["auc"],
        })

        print(
            "Fold {:02d}/{} | Equal {:.4f} | "
            "Adaptive {:.4f} | Tuned {:.4f} | "
            "C={} | Weight={:.2f}".format(
                fold,
                OUTER_SPLITS * OUTER_REPEATS,
                equal_auc,
                adaptive_auc,
                tuned_auc,
                best["c"],
                best["weight"],
            ),
            flush=True,
        )

    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    datasets = load_datasets()
    results = []

    for name, (x, y) in datasets.items():
        result = benchmark_dataset(
            name,
            x,
            y,
        )

        results.append(result)

    combined = pd.concat(
        results,
        ignore_index=True,
    )

    summary_columns = [
        "equal_accuracy",
        "adaptive_accuracy",
        "tuned_accuracy",
        "equal_auc",
        "adaptive_auc",
        "tuned_auc",
        "delta_equal",
        "delta_adaptive",
        "chosen_base_weight",
        "alignment_base_weight",
    ]

    summary = (
        combined.groupby("dataset")[summary_columns]
        .agg(["mean", "std"])
    )

    print(
        "\n--- MULTI-DATASET SVM BENCHMARK ---"
    )

    print(
        summary.round(5).to_string()
    )

    print(
        "\n--- PAIRED ROC-AUC COMPARISONS ---"
    )

    for name, group in combined.groupby("dataset"):
        print("\n{}".format(name))

        for column in (
            "delta_equal",
            "delta_adaptive",
        ):
            differences = group[column].to_numpy()

            wins = int(
                (differences > 1e-12).sum()
            )

            ties = int(
                (np.abs(differences) <= 1e-12).sum()
            )

            losses = int(
                (differences < -1e-12).sum()
            )

            print(
                "{}: mean={:.5f}, "
                "wins/ties/losses={}".format(
                    column,
                    differences.mean(),
                    (wins, ties, losses),
                )
            )

        print(
            "Chosen weights:",
            group["chosen_base_weight"]
            .value_counts()
            .sort_index()
            .to_dict(),
        )

        print(
            "Chosen C:",
            group["chosen_c"]
            .value_counts()
            .sort_index()
            .to_dict(),
        )

    combined.to_csv(
        OUTPUT_DIR / "multidataset_folds.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR / "multidataset_summary.csv",
    )

    print(
        "\nResults saved to {}".format(
            OUTPUT_DIR
        )
    )

    print(
        "Repeated outer folds overlap; "
        "results are descriptive."
    )


if __name__ == "__main__":
    main()
