
"""Compare objective-aware composite SVM selection using nested CV.

Objectives:
    auc:      Maximize mean inner-fold ROC-AUC.
    balanced: Maximize mean inner-fold balanced accuracy.
    hybrid:   Maximize AUC among candidates within 0.02 of
              the best inner-fold balanced accuracy.

Equal-weight and alignment-based SVMs at C=1 are references.
All preprocessing and selection are confined to training folds.
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
HYBRID_TOLERANCE = 0.02

OUTPUT_DIR = Path(__file__).resolve().parent / "results"

METHODS = (
    "equal",
    "alignment",
    "auc",
    "balanced",
    "hybrid",
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
        "Breast Cancer": (cancer.data, cancer.target),
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
                1.0 - float(weight),
            ],
            normalize=False,
        )

    return SVDClassifier(
        C=float(c),
        kernel=kernel,
    )


def measure(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    scores = np.asarray(
        model.decision_function(x_test)
    ).ravel()

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    result = {
        "auc": float(roc_auc_score(y_test, scores)),
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "true_0": int(np.sum(y_test == 0)),
        "true_1": int(np.sum(y_test == 1)),
        "predicted_0": int(np.sum(predictions == 0)),
        "predicted_1": int(np.sum(predictions == 1)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "score_mean": float(np.mean(scores)),
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


def search_candidates(x, y, base, increment, fold):
    """Evaluate candidates using only inner training/validation data."""

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
            balanced_scores = []

            for x_train, y_train, x_valid, y_valid in prepared:
                model = make_model(
                    base,
                    increment,
                    c,
                    weight,
                )

                metrics = measure(
                    model,
                    x_train,
                    y_train,
                    x_valid,
                    y_valid,
                )

                aucs.append(metrics["auc"])
                balanced_scores.append(
                    metrics["balanced_accuracy"]
                )

            candidates.append({
                "c": float(c),
                "weight": float(weight),
                "inner_auc": float(np.mean(aucs)),
                "inner_balanced": float(
                    np.mean(balanced_scores)
                ),
            })

    return candidates


def tie_break(row):
    return (
        row["c"],
        abs(row["weight"] - 0.5),
        row["weight"],
    )


def select_candidate(candidates, objective):
    if objective == "auc":
        return sorted(
            candidates,
            key=lambda row: (
                -row["inner_auc"],
                *tie_break(row)
            ),
        )[0]

    if objective == "balanced":
        return sorted(
            candidates,
            key=lambda row: (
                -row["inner_balanced"],
                -row["inner_auc"],
                *tie_break(row)
            ),
        )[0]

    if objective == "hybrid":
        best_balanced = max(
            row["inner_balanced"]
            for row in candidates
        )

        eligible = [
            row for row in candidates
            if row["inner_balanced"]
            >= best_balanced - HYBRID_TOLERANCE - 1e-12
        ]

        return sorted(
            eligible,
            key=lambda row: (
                -row["inner_auc"],
                -row["inner_balanced"],
                *tie_break(row)
            ),
        )[0]

    raise ValueError(
        "Unknown objective: {}".format(objective)
    )


def benchmark(name, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)

    if set(np.unique(y)) != {0, 1}:
        raise ValueError(
            "{} requires 0/1 labels".format(name)
        )

    midpoint = x.shape[1] // 2
    base = np.arange(midpoint)
    increment = np.arange(midpoint, x.shape[1])

    outer = RepeatedStratifiedKFold(
        n_splits=OUTER_SPLITS,
        n_repeats=OUTER_REPEATS,
        random_state=SEED,
    )

    rows = []

    print(
        "\nDataset: {} | N={} | Features={}".format(
            name, len(y), x.shape[1]
        ),
        flush=True,
    )

    for fold, (train_idx, test_idx) in enumerate(
        outer.split(x, y), start=1
    ):
        x_outer_train = x[train_idx]
        x_outer_test = x[test_idx]
        y_outer_train = y[train_idx]
        y_outer_test = y[test_idx]

        candidates = search_candidates(
            x_outer_train,
            y_outer_train,
            base,
            increment,
            fold,
        )

        selected = {
            objective: select_candidate(
                candidates,
                objective,
            )
            for objective in ("auc", "balanced", "hybrid")
        }

        configurations = {
            "equal": {
                "c": 1.0,
                "weight": 0.5,
            },
            "alignment": {
                "c": 1.0,
                "weight": None,
            },
        }

        configurations.update(selected)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_outer_train)
        x_test = scaler.transform(x_outer_test)

        fold_metrics = {}

        for method in METHODS:
            config = configurations[method]

            model = make_model(
                base,
                increment,
                config["c"],
                config["weight"],
            )

            metrics = measure(
                model,
                x_train,
                y_outer_train,
                x_test,
                y_outer_test,
            )

            if method == "alignment":
                weight = float(
                    model.kernel_.weights_[0]
                )
            else:
                weight = float(config["weight"])

            row = {
                "dataset": name,
                "fold": fold,
                "method": method,
                "c": float(config["c"]),
                "base_weight": weight,
                "inner_auc": config.get(
                    "inner_auc", np.nan
                ),
                "inner_balanced": config.get(
                    "inner_balanced", np.nan
                ),
            }

            row.update(metrics)
            rows.append(row)
            fold_metrics[method] = metrics

        print(
            "Fold {:02d}/{} | "
            "Equal {:.3f}/{:.3f} | "
            "AUC {:.3f}/{:.3f} | "
            "Balanced {:.3f}/{:.3f} | "
            "Hybrid {:.3f}/{:.3f}".format(
                fold,
                OUTER_SPLITS * OUTER_REPEATS,
                fold_metrics["equal"]["auc"],
                fold_metrics["equal"]["balanced_accuracy"],
                fold_metrics["auc"]["auc"],
                fold_metrics["auc"]["balanced_accuracy"],
                fold_metrics["balanced"]["auc"],
                fold_metrics["balanced"]["balanced_accuracy"],
                fold_metrics["hybrid"]["auc"],
                fold_metrics["hybrid"]["balanced_accuracy"],
            ),
            flush=True,
        )

    return pd.DataFrame(rows)


def summarize(results):
    columns = [
        "auc",
        "accuracy",
        "balanced_accuracy",
    ]

    summary = (
        results.groupby(["dataset", "method"])[columns]
        .agg(["mean", "std"])
    )

    print("\n--- OBJECTIVE COMPARISON ---")
    print(summary.round(5).to_string())

    print("\n--- PAIRED DIFFERENCES VS EQUAL ---")

    for dataset, group in results.groupby("dataset"):
        print("\n{}".format(dataset))

        for metric in ("auc", "balanced_accuracy"):
            pivot = group.pivot(
                index="fold",
                columns="method",
                values=metric,
            )

            for method in (
                "alignment",
                "auc",
                "balanced",
                "hybrid",
            ):
                delta = (
                    pivot[method] - pivot["equal"]
                ).to_numpy()

                wins = int((delta > 1e-12).sum())
                ties = int(
                    (np.abs(delta) <= 1e-12).sum()
                )
                losses = int((delta < -1e-12).sum())

                print(
                    "{} | {} | delta={:+.5f} | "
                    "W/T/L={}".format(
                        metric,
                        method,
                        np.mean(delta),
                        (wins, ties, losses),
                    )
                )

    print("\n--- SELECTED HYPERPARAMETERS ---")

    for (dataset, method), group in results.groupby(
        ["dataset", "method"]
    ):
        if method not in ("auc", "balanced", "hybrid"):
            continue

        print(
            "{} | {} | C={} | weights={}".format(
                dataset,
                method,
                group["c"].value_counts()
                .sort_index().to_dict(),
                group["base_weight"].value_counts()
                .sort_index().to_dict(),
            )
        )

    print("\n--- PREDICTION DIAGNOSTICS ---")

    diagnostics = (
        results.groupby(["dataset", "method"])[
            [
                "predicted_0",
                "predicted_1",
                "score_min",
                "score_max",
                "score_mean_0",
                "score_mean_1",
            ]
        ]
        .mean()
    )

    print(diagnostics.round(4).to_string())

    return summary


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []

    for name, (x, y) in load_datasets().items():
        all_results.append(
            benchmark(name, x, y)
        )

    results = pd.concat(
        all_results,
        ignore_index=True,
    )

    summary = summarize(results)

    results.to_csv(
        OUTPUT_DIR / "objective_comparison_folds.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR / "objective_comparison_summary.csv",
    )

    print(
        "\nResults saved to {}".format(
            OUTPUT_DIR
        )
    )
    print(
        "Repeated outer folds overlap; "
        "comparisons are descriptive."
    )


if __name__ == "__main__":
    main()
