
"""Nested-CV ablation of SVM kernel weights and regularization.

Models:
    equal:       fixed C=1, fixed base weight=0.5
    c_only:      tuned C, fixed base weight=0.5
    weight_only: fixed C=1, tuned base weight
    joint:       tuned C and base weight
    alignment:   fixed C=1, kernel-target alignment weights

All hyperparameter selection uses inner training folds only.
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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
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


SEED = 42
GAMMA = 0.5

OUTER_SPLITS = 5
OUTER_REPEATS = 2
INNER_SPLITS = 3

C_VALUES = (0.1, 1.0, 10.0)
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)

OUTPUT_DIR = (
    Path(__file__).resolve().parent / "results"
)

MODEL_NAMES = (
    "equal",
    "c_only",
    "weight_only",
    "joint",
    "alignment",
)


def load_datasets():
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
        "Moons": (moons_x, moons_y),
        "Circles": (circles_x, circles_y),
    }


def group_kernel(indices):
    indices = np.asarray(indices, dtype=int)

    def kernel(x, y):
        return rbf_kernel(
            np.asarray(x)[:, indices],
            np.asarray(y)[:, indices],
            gamma=GAMMA,
        )

    return kernel


def make_model(base, increment, c, weight=None):
    kernels = [
        group_kernel(base),
        group_kernel(increment),
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


def fit_and_measure(
    model,
    x_train,
    y_train,
    x_test,
    y_test,
):
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    scores = model.decision_function(x_test)

    labels = np.array([0, 1])

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    tn, fp, fn, tp = matrix.ravel()

    result = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "auc": float(
            roc_auc_score(y_test, scores)
        ),
        "predicted_0": int(
            np.sum(predictions == 0)
        ),
        "predicted_1": int(
            np.sum(predictions == 1)
        ),
        "true_0": int(
            np.sum(y_test == 0)
        ),
        "true_1": int(
            np.sum(y_test == 1)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "score_mean_0": float(
            np.mean(scores[y_test == 0])
        ),
        "score_mean_1": float(
            np.mean(scores[y_test == 1])
        ),
    }

    return result


def inner_search(x, y, base, increment, fold):
    """Evaluate every C/weight combination on identical inner folds."""

    cv = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=SEED + fold,
    )

    prepared = []

    for train_idx, valid_idx in cv.split(x, y):
        scaler = StandardScaler()

        x_train = scaler.fit_transform(
            x[train_idx]
        )
        x_valid = scaler.transform(
            x[valid_idx]
        )

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

                model.fit(x_train, y_train)

                scores = model.decision_function(
                    x_valid
                )

                aucs.append(
                    roc_auc_score(y_valid, scores)
                )

            candidates.append({
                "c": float(c),
                "weight": float(weight),
                "inner_auc": float(np.mean(aucs)),
            })

    return candidates


def choose_candidate(candidates, mode):
    if mode == "c_only":
        eligible = [
            row for row in candidates
            if row["weight"] == 0.5
        ]
    elif mode == "weight_only":
        eligible = [
            row for row in candidates
            if row["c"] == 1.0
        ]
    elif mode == "joint":
        eligible = candidates
    else:
        raise ValueError(
            "Unknown tuning mode: {}".format(mode)
        )

    return sorted(
        eligible,
        key=lambda row: (
            -row["inner_auc"],
            row["c"],
            abs(row["weight"] - 0.5),
            row["weight"],
        ),
    )[0]


def benchmark_dataset(name, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)

    if set(np.unique(y)) != {0, 1}:
        raise ValueError(
            "{} requires binary 0/1 labels".format(name)
        )

    midpoint = x.shape[1] // 2
    base = np.arange(midpoint)
    increment = np.arange(
        midpoint,
        x.shape[1],
    )

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
            x.shape[1],
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

        candidates = inner_search(
            x_outer_train,
            y_outer_train,
            base,
            increment,
            fold,
        )

        selected = {
            mode: choose_candidate(
                candidates,
                mode,
            )
            for mode in (
                "c_only",
                "weight_only",
                "joint",
            )
        }

        configurations = {
            "equal": {
                "c": 1.0,
                "weight": 0.5,
            },
            "c_only": selected["c_only"],
            "weight_only": selected["weight_only"],
            "joint": selected["joint"],
            "alignment": {
                "c": 1.0,
                "weight": None,
            },
        }

        scaler = StandardScaler()

        x_train = scaler.fit_transform(
            x_outer_train
        )
        x_test = scaler.transform(
            x_outer_test
        )

        fold_results = {}

        for mode in MODEL_NAMES:
            config = configurations[mode]

            model = make_model(
                base,
                increment,
                config["c"],
                config["weight"],
            )

            measured = fit_and_measure(
                model,
                x_train,
                y_outer_train,
                x_test,
                y_outer_test,
            )

            if mode == "alignment":
                learned_weight = float(
                    model.kernel_.weights_[0]
                )
            else:
                learned_weight = float(
                    config["weight"]
                )

            row = {
                "dataset": name,
                "fold": fold,
                "model": mode,
                "n_samples": len(y),
                "n_features": x.shape[1],
                "c": float(config["c"]),
                "base_weight": learned_weight,
                "increment_weight": 1.0 - learned_weight,
                "inner_auc": (
                    config.get("inner_auc", np.nan)
                ),
            }

            row.update(measured)
            rows.append(row)
            fold_results[mode] = measured

        print(
            "Fold {:02d}/{} | Equal {:.4f} | "
            "C-only {:.4f} | Weight-only {:.4f} | "
            "Joint {:.4f} | Alignment {:.4f}".format(
                fold,
                OUTER_SPLITS * OUTER_REPEATS,
                fold_results["equal"]["auc"],
                fold_results["c_only"]["auc"],
                fold_results["weight_only"]["auc"],
                fold_results["joint"]["auc"],
                fold_results["alignment"]["auc"],
            ),
            flush=True,
        )

    return pd.DataFrame(rows)


def print_summary(results):
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "auc",
    ]

    summary = (
        results.groupby(
            ["dataset", "model"]
        )[metrics]
        .agg(["mean", "std"])
    )

    print("\n--- ABLATION SUMMARY ---")
    print(summary.round(5).to_string())

    print("\n--- PAIRED AUC DIFFERENCES VS EQUAL ---")

    for name, group in results.groupby("dataset"):
        pivot = group.pivot(
            index="fold",
            columns="model",
            values="auc",
        )

        print("\n{}".format(name))

        for mode in (
            "c_only",
            "weight_only",
            "joint",
            "alignment",
        ):
            delta = (
                pivot[mode] - pivot["equal"]
            ).to_numpy()

            wins = int((delta > 1e-12).sum())
            ties = int(
                (np.abs(delta) <= 1e-12).sum()
            )
            losses = int((delta < -1e-12).sum())

            print(
                "{}: delta={:+.5f}, W/T/L={}".format(
                    mode,
                    np.mean(delta),
                    (wins, ties, losses),
                )
            )

    print("\n--- SELECTED HYPERPARAMETERS ---")

    for (name, mode), group in results.groupby(
        ["dataset", "model"]
    ):
        print(
            "{} | {} | C={} | Base weights={}".format(
                name,
                mode,
                group["c"].value_counts()
                .sort_index().to_dict(),
                group["base_weight"].round(2)
                .value_counts().sort_index().to_dict(),
            )
        )

    return summary


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []

    for name, (x, y) in load_datasets().items():
        all_results.append(
            benchmark_dataset(name, x, y)
        )

    results = pd.concat(
        all_results,
        ignore_index=True,
    )

    summary = print_summary(results)

    results.to_csv(
        OUTPUT_DIR / "multidataset_ablation_folds.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR / "multidataset_ablation_summary.csv",
    )

    print(
        "\nSaved results to {}".format(
            OUTPUT_DIR
        )
    )

    print(
        "Outer folds overlap across repeats; "
        "results are descriptive."
    )


if __name__ == "__main__":
    main()
