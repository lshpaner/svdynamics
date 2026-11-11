
"""Nested-CV comparison of composite SVMs and conventional RBF SVC.

The hybrid composite selects C and kernel weight using an
inner-CV balanced-accuracy safeguard. The tuned RBF SVC selects
C and gamma using the same safeguard.

Results are exploratory. Repeated outer folds overlap.
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
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
RBF_GAMMAS = ("scale", 0.01, 0.1, 0.5, 1.0)

COMPOSITE_GAMMA = 0.5
BALANCED_TOLERANCE = 0.02

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MAGIC_PATH = ROOT / "data" / "magic_sample.csv"


def datasets():
    cancer = load_breast_cancer()
    digits = load_digits()

    digit_mask = np.isin(digits.target, [3, 8])

    x_moons, y_moons = make_moons(
        n_samples=600,
        noise=0.25,
        random_state=SEED,
    )

    x_circles, y_circles = make_circles(
        n_samples=600,
        noise=0.12,
        factor=0.45,
        random_state=SEED,
    )

    data = {
        "Breast Cancer": (
            cancer.data,
            cancer.target,
        ),
        "Digits 3 vs 8": (
            digits.data[digit_mask],
            (digits.target[digit_mask] == 8).astype(int),
        ),
        "Moons": (x_moons, y_moons),
        "Circles": (x_circles, y_circles),
    }

    if MAGIC_PATH.exists():
        frame = pd.read_csv(MAGIC_PATH)

        target_candidates = [
            "target", "class", "label", "Class", "Target"
        ]

        target_col = next(
            (
                col for col in target_candidates
                if col in frame.columns
            ),
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

        features = frame.drop(columns=[target_col])

        # The MAGIC dataset is expected to contain 10 numeric predictors.
        features = features.select_dtypes(include=[np.number])

        if features.shape[1] != 10:
            raise ValueError(
                "Expected 10 numeric MAGIC features, found {}. "
                "Check target_col and CSV schema.".format(
                    features.shape[1]
                )
            )

        if features.isna().any().any():
            raise ValueError("MAGIC features contain missing values")

        data["MAGIC"] = (
            features.to_numpy(dtype=float),
            y,
        )
    else:
        print(
            "MAGIC CSV not found; skipping MAGIC: {}".format(
                MAGIC_PATH
            ),
            flush=True,
        )

    return data


def make_group_kernel(indices):
    indices = np.asarray(indices, dtype=int)

    def kernel(x, y):
        return rbf_kernel(
            np.asarray(x)[:, indices],
            np.asarray(y)[:, indices],
            gamma=COMPOSITE_GAMMA,
        )

    return kernel


def composite_model(base, increment, c, weight):
    kernel = CompositeKernel(
        kernels=[
            make_group_kernel(base),
            make_group_kernel(increment),
        ],
        weights=[float(weight), float(1.0 - weight)],
        normalize=False,
    )

    return SVDClassifier(
        C=float(c),
        kernel=kernel,
    )


def rbf_model(c, gamma):
    return SVC(
        C=float(c),
        kernel="rbf",
        gamma=gamma,
    )


def evaluate(model, x_train, y_train, x_valid, y_valid):
    model.fit(x_train, y_train)

    scores = np.asarray(
        model.decision_function(x_valid)
    ).ravel()

    predictions = model.predict(x_valid)

    return {
        "auc": float(roc_auc_score(y_valid, scores)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_valid, predictions)
        ),
        "accuracy": float(
            accuracy_score(y_valid, predictions)
        ),
        "predicted_0": int(np.sum(predictions == 0)),
        "predicted_1": int(np.sum(predictions == 1)),
    }


def inner_folds(x, y, fold_number):
    cv = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=SEED + fold_number,
    )

    prepared = []

    for train_idx, valid_idx in cv.split(x, y):
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


def score_candidate(builder, folds):
    metrics = [
        evaluate(
            builder(),
            x_train,
            y_train,
            x_valid,
            y_valid,
        )
        for x_train, y_train, x_valid, y_valid in folds
    ]

    return {
        "inner_auc": float(
            np.mean([m["auc"] for m in metrics])
        ),
        "inner_balanced": float(
            np.mean([m["balanced_accuracy"] for m in metrics])
        ),
    }


def hybrid_select(candidates):
    best_balanced = max(
        row["inner_balanced"]
        for row in candidates
    )

    eligible = [
        row for row in candidates
        if row["inner_balanced"]
        >= best_balanced - BALANCED_TOLERANCE - 1e-12
    ]

    return sorted(
        eligible,
        key=lambda row: (
            -row["inner_auc"],
            -row["inner_balanced"],
            row["c"],
            row["tie_break"],
        ),
    )[0]


def select_composite(folds, base, increment):
    candidates = []

    for c in C_VALUES:
        for weight in WEIGHTS:
            metrics = score_candidate(
                lambda c=c, weight=weight: composite_model(
                    base, increment, c, weight
                ),
                folds,
            )

            candidates.append({
                "c": float(c),
                "weight": float(weight),
                "tie_break": abs(weight - 0.5),
                **metrics,
            })

    return hybrid_select(candidates)


def select_rbf(folds):
    candidates = []

    for c in C_VALUES:
        for gamma_idx, gamma in enumerate(RBF_GAMMAS):
            metrics = score_candidate(
                lambda c=c, gamma=gamma: rbf_model(
                    c, gamma
                ),
                folds,
            )

            candidates.append({
                "c": float(c),
                "gamma": gamma,
                "tie_break": gamma_idx,
                **metrics,
            })

    return hybrid_select(candidates)


def benchmark(name, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)

    if set(np.unique(y)) != {0, 1}:
        raise ValueError(
            "{} must have binary 0/1 labels".format(name)
        )

    midpoint = x.shape[1] // 2
    base = np.arange(midpoint)
    increment = np.arange(midpoint, x.shape[1])

    outer = RepeatedStratifiedKFold(
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
        outer.split(x, y), start=1
    ):
        x_outer_train = x[train_idx]
        y_outer_train = y[train_idx]
        x_outer_test = x[test_idx]
        y_outer_test = y[test_idx]

        prepared = inner_folds(
            x_outer_train,
            y_outer_train,
            fold,
        )

        chosen_composite = select_composite(
            prepared, base, increment
        )

        chosen_rbf = select_rbf(prepared)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_outer_train)
        x_test = scaler.transform(x_outer_test)

        models = {
            "composite_equal": composite_model(
                base, increment, 1.0, 0.5
            ),
            "composite_hybrid": composite_model(
                base,
                increment,
                chosen_composite["c"],
                chosen_composite["weight"],
            ),
            "rbf_fixed": rbf_model(1.0, "scale"),
            "rbf_tuned": rbf_model(
                chosen_rbf["c"],
                chosen_rbf["gamma"],
            ),
        }

        fold_results = {}

        for method, model in models.items():
            metrics = evaluate(
                model,
                x_train,
                y_outer_train,
                x_test,
                y_outer_test,
            )

            fold_results[method] = metrics

            if method == "composite_hybrid":
                c = chosen_composite["c"]
                weight = chosen_composite["weight"]
                gamma = COMPOSITE_GAMMA
            elif method == "composite_equal":
                c = 1.0
                weight = 0.5
                gamma = COMPOSITE_GAMMA
            elif method == "rbf_tuned":
                c = chosen_rbf["c"]
                weight = np.nan
                gamma = chosen_rbf["gamma"]
            else:
                c = 1.0
                weight = np.nan
                gamma = "scale"

            rows.append({
                "dataset": name,
                "fold": fold,
                "method": method,
                "c": c,
                "weight": weight,
                "gamma": gamma,
                **metrics,
            })

        print(
            "Fold {:02d}/10 | "
            "Composite {:.4f} | "
            "RBF tuned {:.4f} | "
            "Delta {:+.4f}".format(
                fold,
                fold_results["composite_hybrid"]["auc"],
                fold_results["rbf_tuned"]["auc"],
                fold_results["composite_hybrid"]["auc"]
                - fold_results["rbf_tuned"]["auc"],
            ),
            flush=True,
        )

    return pd.DataFrame(rows)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    frames = []

    for name, (x, y) in datasets().items():
        frames.append(benchmark(name, x, y))

    results = pd.concat(frames, ignore_index=True)

    summary = results.groupby(
        ["dataset", "method"]
    )[["auc", "accuracy", "balanced_accuracy"]].agg(
        ["mean", "std"]
    )

    print("\n--- RBF VS COMPOSITE BENCHMARK ---")
    print(summary.round(5).to_string())

    print("\n--- PAIRED COMPOSITE VS TUNED RBF ---")

    for dataset, group in results.groupby("dataset"):
        for metric in ("auc", "balanced_accuracy"):
            pivot = group.pivot(
                index="fold",
                columns="method",
                values=metric,
            )

            delta = (
                pivot["composite_hybrid"]
                - pivot["rbf_tuned"]
            ).to_numpy()

            wins = int((delta > 1e-12).sum())
            ties = int((np.abs(delta) <= 1e-12).sum())
            losses = int((delta < -1e-12).sum())

            print(
                "{} | {} | Delta={:+.5f} | W/T/L={}".format(
                    dataset,
                    metric,
                    np.mean(delta),
                    (wins, ties, losses),
                )
            )

    print("\n--- TUNED HYPERPARAMETERS ---")

    for (dataset, method), group in results.groupby(
        ["dataset", "method"]
    ):
        if method not in ("composite_hybrid", "rbf_tuned"):
            continue

        print(
            "{} | {} | C={} | gamma={} | weights={}".format(
                dataset,
                method,
                group["c"].value_counts().sort_index().to_dict(),
                group["gamma"].value_counts().to_dict(),
                group["weight"].dropna()
                .value_counts().sort_index().to_dict(),
            )
        )

    results.to_csv(
        RESULTS / "rbf_comparison_folds.csv",
        index=False,
    )
    summary.to_csv(
        RESULTS / "rbf_comparison_summary.csv"
    )

    print("\nResults saved to {}".format(RESULTS))
    print(
        "Repeated outer folds overlap; results are descriptive."
    )


if __name__ == "__main__":
    main()
