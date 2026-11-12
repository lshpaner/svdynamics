
"""Controlled comparison: RBF vs equal and tuned composite kernels.

All models:
- Use identical nested CV splits and train-only standardization.
- Tune C and gamma on the same grid.
- Use the same balanced-accuracy-constrained AUC objective.

The tuned composite additionally searches base-kernel weight.
Component kernels share one gamma. Results are descriptive.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    make_circles,
    make_moons,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from svdynamics import CompositeKernel, SVDClassifier


SEED = 42
OUTER_SPLITS = 5
OUTER_REPEATS = 2
INNER_SPLITS = 3

C_VALUES = (0.1, 1.0, 10.0)
GAMMAS = ("scale", 0.01, 0.1, 0.5, 1.0)
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
BALANCED_TOLERANCE = 0.02

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MAGIC_PATH = ROOT / "data" / "magic_sample.csv"

METHODS = ("rbf", "equal", "adaptive")


def load_datasets():
    cancer = load_breast_cancer()
    digits = load_digits()

    mask = np.isin(digits.target, [3, 8])

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

    datasets = {
        "Breast Cancer": (
            cancer.data,
            cancer.target,
        ),
        "Digits 3 vs 8": (
            digits.data[mask],
            (digits.target[mask] == 8).astype(int),
        ),
        "Moons": (moons_x, moons_y),
        "Circles": (circles_x, circles_y),
    }

    if MAGIC_PATH.exists():
        frame = pd.read_csv(MAGIC_PATH)

        possible_targets = (
            "target", "class", "label", "Class", "Target"
        )

        target_col = next(
            (c for c in possible_targets if c in frame),
            frame.columns[-1],
        )

        target = frame[target_col]

        if target.isna().any():
            raise ValueError("MAGIC target contains missing values")

        unique = set(target.unique())

        if unique == {-1, 1}:
            y = (target.to_numpy() == 1).astype(int)
        elif unique == {0, 1}:
            y = target.to_numpy().astype(int)
        elif unique == {"g", "h"}:
            y = (target.to_numpy() == "g").astype(int)
        else:
            raise ValueError(
                "Unexpected MAGIC target values: {}".format(unique)
            )

        features = (
            frame.drop(columns=[target_col])
            .select_dtypes(include=[np.number])
        )

        if features.shape[1] != 10:
            raise ValueError(
                "Expected 10 MAGIC features, found {}".format(
                    features.shape[1]
                )
            )

        if features.isna().any().any():
            raise ValueError("MAGIC features contain missing values")

        datasets["MAGIC"] = (
            features.to_numpy(dtype=float),
            y,
        )
    else:
        print(
            "MAGIC missing, skipping: {}".format(MAGIC_PATH),
            flush=True,
        )

    return datasets


def numeric_gamma(gamma, x_train):
    """Match sklearn's gamma='scale' on standardized training data."""
    if gamma == "scale":
        variance = float(np.asarray(x_train).var())
        if variance <= 0:
            raise ValueError("Zero training-data variance")
        return 1.0 / (x_train.shape[1] * variance)

    return float(gamma)


def group_kernel(indices, gamma):
    indices = np.asarray(indices, dtype=int)

    def kernel(x, y):
        return rbf_kernel(
            np.asarray(x)[:, indices],
            np.asarray(y)[:, indices],
            gamma=gamma,
        )

    return kernel


def build_model(method, base, increment, c, gamma, weight,
                x_train):
    if method == "rbf":
        return SVC(
            C=float(c),
            kernel="rbf",
            gamma=gamma,
        )

    resolved_gamma = numeric_gamma(gamma, x_train)

    kernel = CompositeKernel(
        kernels=[
            group_kernel(base, resolved_gamma),
            group_kernel(increment, resolved_gamma),
        ],
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


def evaluate(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    scores = np.asarray(
        model.decision_function(x_test)
    ).ravel()

    return {
        "auc": float(roc_auc_score(y_test, scores)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "predicted_0": int(np.sum(predictions == 0)),
        "predicted_1": int(np.sum(predictions == 1)),
    }


def prepare_inner_folds(x, y, fold_number):
    splitter = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=SEED + fold_number,
    )

    prepared = []

    for train_idx, valid_idx in splitter.split(x, y):
        scaler = StandardScaler()

        x_train = scaler.fit_transform(x[train_idx])
        x_valid = scaler.transform(x[valid_idx])

        prepared.append((
            x_train,
            y[train_idx],
            x_valid,
            y[valid_idx],
        ))

    return prepared


def select_candidate(method, folds, base, increment):
    candidates = []

    weights = (
        WEIGHTS if method == "adaptive" else (0.5,)
    )

    for c in C_VALUES:
        for gamma in GAMMAS:
            for weight in weights:
                metrics = []

                for x_train, y_train, x_valid, y_valid in folds:
                    model = build_model(
                        method,
                        base,
                        increment,
                        c,
                        gamma,
                        weight,
                        x_train,
                    )

                    metrics.append(
                        evaluate(
                            model,
                            x_train,
                            y_train,
                            x_valid,
                            y_valid,
                        )
                    )

                candidates.append({
                    "c": float(c),
                    "gamma": gamma,
                    "weight": float(weight),
                    "inner_auc": float(np.mean([
                        m["auc"] for m in metrics
                    ])),
                    "inner_balanced": float(np.mean([
                        m["balanced_accuracy"]
                        for m in metrics
                    ])),
                })

    best_balanced = max(
        candidate["inner_balanced"]
        for candidate in candidates
    )

    eligible = [
        candidate
        for candidate in candidates
        if candidate["inner_balanced"]
        >= best_balanced - BALANCED_TOLERANCE - 1e-12
    ]

    gamma_order = {
        value: index
        for index, value in enumerate(GAMMAS)
    }

    return sorted(
        eligible,
        key=lambda row: (
            -row["inner_auc"],
            -row["inner_balanced"],
            row["c"],
            gamma_order[row["gamma"]],
            abs(row["weight"] - 0.5),
            row["weight"],
        ),
    )[0]


def benchmark(name, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)

    if set(np.unique(y)) != {0, 1}:
        raise ValueError(
            "{} requires binary 0/1 labels".format(name)
        )

    midpoint = x.shape[1] // 2
    base = np.arange(midpoint)
    increment = np.arange(midpoint, x.shape[1])

    splitter = RepeatedStratifiedKFold(
        n_splits=OUTER_SPLITS,
        n_repeats=OUTER_REPEATS,
        random_state=SEED,
    )

    print(
        "\nDataset: {} | N={} | Features={}".format(
            name, len(y), x.shape[1]
        ),
        flush=True,
    )

    rows = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(x, y),
        start=1,
    ):
        x_outer_train = x[train_idx]
        y_outer_train = y[train_idx]
        x_outer_test = x[test_idx]
        y_outer_test = y[test_idx]

        inner_folds = prepare_inner_folds(
            x_outer_train,
            y_outer_train,
            fold,
        )

        chosen = {
            method: select_candidate(
                method,
                inner_folds,
                base,
                increment,
            )
            for method in METHODS
        }

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_outer_train)
        x_test = scaler.transform(x_outer_test)

        fold_results = {}

        for method in METHODS:
            params = chosen[method]

            model = build_model(
                method,
                base,
                increment,
                params["c"],
                params["gamma"],
                params["weight"],
                x_train,
            )

            metrics = evaluate(
                model,
                x_train,
                y_outer_train,
                x_test,
                y_outer_test,
            )

            fold_results[method] = metrics

            rows.append({
                "dataset": name,
                "fold": fold,
                "method": method,
                "c": params["c"],
                "gamma": params["gamma"],
                "weight": (
                    params["weight"]
                    if method != "rbf" else np.nan
                ),
                "inner_auc": params["inner_auc"],
                "inner_balanced": params["inner_balanced"],
                **metrics,
            })

        print(
            "Fold {:02d}/10 | "
            "RBF {:.4f} | "
            "Equal {:.4f} | "
            "Adaptive {:.4f} | "
            "Adaptive-RBF {:+.4f}".format(
                fold,
                fold_results["rbf"]["auc"],
                fold_results["equal"]["auc"],
                fold_results["adaptive"]["auc"],
                fold_results["adaptive"]["auc"]
                - fold_results["rbf"]["auc"],
            ),
            flush=True,
        )

    return pd.DataFrame(rows)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    frames = [
        benchmark(name, x, y)
        for name, (x, y) in load_datasets().items()
    ]

    results = pd.concat(frames, ignore_index=True)

    metrics = ["auc", "accuracy", "balanced_accuracy"]

    summary = (
        results.groupby(["dataset", "method"])[metrics]
        .agg(["mean", "std"])
    )

    print("\n--- CONTROLLED KERNEL COMPARISON ---")
    print(summary.round(5).to_string())

    print("\n--- PAIRED DIFFERENCES ---")

    for dataset, group in results.groupby("dataset"):
        print("\n{}".format(dataset))

        for metric in ("auc", "balanced_accuracy"):
            pivot = group.pivot(
                index="fold",
                columns="method",
                values=metric,
            )

            for first, second in (
                ("adaptive", "rbf"),
                ("adaptive", "equal"),
                ("equal", "rbf"),
            ):
                delta = (
                    pivot[first] - pivot[second]
                ).to_numpy()

                wins = int(np.sum(delta > 1e-12))
                ties = int(np.sum(np.abs(delta) <= 1e-12))
                losses = int(np.sum(delta < -1e-12))

                print(
                    "{} | {} vs {} | delta={:+.5f} "
                    "| W/T/L={}".format(
                        metric,
                        first,
                        second,
                        float(np.mean(delta)),
                        (wins, ties, losses),
                    )
                )

    print("\n--- SELECTED HYPERPARAMETERS ---")

    for (dataset, method), group in results.groupby(
        ["dataset", "method"]
    ):
        print(
            "{} | {} | C={} | gamma={} | weights={}".format(
                dataset,
                method,
                group["c"].value_counts()
                .sort_index().to_dict(),
                group["gamma"].value_counts().to_dict(),
                group["weight"].dropna()
                .value_counts().sort_index().to_dict(),
            )
        )

    results.to_csv(
        RESULTS / "controlled_kernel_folds.csv",
        index=False,
    )

    summary.to_csv(
        RESULTS / "controlled_kernel_summary.csv",
    )

    print("\nSaved results to {}".format(RESULTS))
    print(
        "Repeated outer folds overlap; "
        "results are descriptive."
    )


if __name__ == "__main__":
    main()
