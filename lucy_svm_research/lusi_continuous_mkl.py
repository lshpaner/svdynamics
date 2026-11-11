
"""Continuous kernel-weight optimization for composite SVMs.

Compare:
    equal:      fixed 0.5 / 0.5 weights
    alignment:  existing kernel-target alignment weights
    grid:       discrete validation-tuned weights
    continuous: bounded continuous optimization of ranking loss

All model selection occurs inside nested CV. Preprocessing is
fitted only on the corresponding training partition.

The continuous optimizer minimizes pairwise logistic ranking
loss on inner-validation scores. This is NOT joint-margin MKL.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import minimize_scalar
from scipy.special import log_expit

from sklearn.datasets import make_moons
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

from svdynamics import (
    AdaptiveCompositeKernel,
    CompositeKernel,
    SVDClassifier,
)


SEED = 42
OUTER_SPLITS = 5
OUTER_REPEATS = 2
INNER_SPLITS = 3

C_VALUES = (1.0, 10.0)
GAMMAS = ("scale", 0.1, 0.5)
GRID_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)

BALANCED_TOLERANCE = 0.02

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
MAGIC_PATH = ROOT / "data" / "magic_sample.csv"


def load_data():
    moons_x, moons_y = make_moons(
        n_samples=600,
        noise=0.25,
        random_state=SEED,
    )

    datasets = {
        "Moons": (moons_x, moons_y),
    }

    if MAGIC_PATH.exists():
        frame = pd.read_csv(MAGIC_PATH)

        target_names = (
            "target", "class", "label", "Class", "Target"
        )
        target_col = next(
            (col for col in target_names if col in frame.columns),
            frame.columns[-1],
        )

        target = frame[target_col]
        unique = set(target.dropna().unique())

        if target.isna().any():
            raise ValueError("MAGIC target contains missing values")

        if unique == {-1, 1}:
            y = (target.to_numpy() == 1).astype(int)
        elif unique == {0, 1}:
            y = target.to_numpy().astype(int)
        elif unique == {"g", "h"}:
            y = (target.to_numpy() == "g").astype(int)
        else:
            raise ValueError(
                "Unexpected MAGIC labels: {}".format(unique)
            )

        features = (
            frame.drop(columns=[target_col])
            .select_dtypes(include=[np.number])
        )

        if features.shape[1] != 10:
            raise ValueError(
                "Expected 10 MAGIC predictors, found {}".format(
                    features.shape[1]
                )
            )

        if features.isna().any().any():
            raise ValueError("MAGIC contains missing predictors")

        datasets["MAGIC"] = (
            features.to_numpy(dtype=float),
            y,
        )
    else:
        print("MAGIC CSV missing, running Moons only.")

    return datasets


def resolve_gamma(gamma, x_train):
    if gamma == "scale":
        variance = float(np.asarray(x_train).var())

        if variance <= 0:
            raise ValueError("Training variance must be positive")

        return 1.0 / (x_train.shape[1] * variance)

    return float(gamma)


def group_kernel(columns, gamma):
    columns = np.asarray(columns, dtype=int)

    def kernel(x, y):
        return rbf_kernel(
            np.asarray(x)[:, columns],
            np.asarray(y)[:, columns],
            gamma=gamma,
        )

    return kernel


def build_model(
    method, base, increment, c, gamma, weight, x_train
):
    gamma_value = resolve_gamma(gamma, x_train)

    kernels = [
        group_kernel(base, gamma_value),
        group_kernel(increment, gamma_value),
    ]

    if method == "alignment":
        kernel = AdaptiveCompositeKernel(
            kernels=kernels,
            normalize=False,
        )
    else:
        kernel = CompositeKernel(
            kernels=kernels,
            weights=[float(weight), 1.0 - float(weight)],
            normalize=False,
        )

    return SVDClassifier(
        C=float(c),
        kernel=kernel,
    )


def evaluate(
    method, base, increment, c, gamma, weight,
    x_train, y_train, x_valid, y_valid
):
    model = build_model(
        method, base, increment, c, gamma, weight, x_train
    )
    model.fit(x_train, y_train)

    scores = np.asarray(
        model.decision_function(x_valid)
    ).ravel()
    predictions = model.predict(x_valid)

    output = {
        "auc": float(roc_auc_score(y_valid, scores)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_valid, predictions)
        ),
        "accuracy": float(
            accuracy_score(y_valid, predictions)
        ),
        "scores": scores,
    }

    if method == "alignment":
        output["weight"] = float(model.kernel_.weights_[0])
    else:
        output["weight"] = float(weight)

    return output


def ranking_loss(scores, y):
    """Pairwise logistic ranking loss; smaller is better."""
    positive = scores[y == 1]
    negative = scores[y == 0]

    differences = (
        positive[:, None] - negative[None, :]
    )

    return float(
        -np.mean(log_expit(differences))
    )


def prepare_inner(x, y, fold):
    splitter = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=SEED + fold,
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


def candidate_metrics(
    method, folds, base, increment, c, gamma, weight
):
    evaluated = [
        evaluate(
            method,
            base,
            increment,
            c,
            gamma,
            weight,
            x_train,
            y_train,
            x_valid,
            y_valid,
        )
        for x_train, y_train, x_valid, y_valid in folds
    ]

    return {
        "c": float(c),
        "gamma": gamma,
        "weight": float(
            np.mean([m["weight"] for m in evaluated])
        ),
        "inner_auc": float(
            np.mean([m["auc"] for m in evaluated])
        ),
        "inner_balanced": float(
            np.mean([
                m["balanced_accuracy"] for m in evaluated
            ])
        ),
        "inner_ranking_loss": float(
            np.mean([
                ranking_loss(m["scores"], y_valid)
                for m, (_, _, _, y_valid)
                in zip(evaluated, folds)
            ])
        ),
    }


def select_hybrid(candidates):
    best_balanced = max(
        row["inner_balanced"] for row in candidates
    )

    eligible = [
        row for row in candidates
        if row["inner_balanced"]
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
        ),
    )[0]


def select_model(method, folds, base, increment):
    candidates = []

    for c in C_VALUES:
        for gamma in GAMMAS:
            if method == "equal":
                weights = [0.5]

            elif method == "alignment":
                weights = [None]

            elif method == "grid":
                weights = GRID_WEIGHTS

            elif method == "continuous":

                def objective(weight):
                    result = candidate_metrics(
                        "composite",
                        folds,
                        base,
                        increment,
                        c,
                        gamma,
                        weight,
                    )
                    return result["inner_ranking_loss"]

                optimization = minimize_scalar(
                    objective,
                    bounds=(0.0, 1.0),
                    method="bounded",
                    options={
                        "xatol": 0.01,
                        "maxiter": 20,
                    },
                )

                weights = [
                    0.0,
                    float(optimization.x),
                    1.0,
                ]

            else:
                raise ValueError(
                    "Unknown method: {}".format(method)
                )

            for weight in weights:
                candidates.append(
                    candidate_metrics(
                        (
                            "alignment"
                            if method == "alignment"
                            else "composite"
                        ),
                        folds,
                        base,
                        increment,
                        c,
                        gamma,
                        weight,
                    )
                )

    return select_hybrid(candidates)


def benchmark(name, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)

    midpoint = x.shape[1] // 2
    base = np.arange(midpoint)
    increment = np.arange(midpoint, x.shape[1])

    splitter = RepeatedStratifiedKFold(
        n_splits=OUTER_SPLITS,
        n_repeats=OUTER_REPEATS,
        random_state=SEED,
    )

    methods = ("equal", "alignment", "grid", "continuous")
    rows = []

    print(
        "\nDataset: {} | N={} | Features={}".format(
            name, len(y), x.shape[1]
        ),
        flush=True,
    )

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(x, y), start=1
    ):
        x_outer_train = x[train_idx]
        y_outer_train = y[train_idx]

        folds = prepare_inner(
            x_outer_train, y_outer_train, fold
        )

        selected = {
            method: select_model(
                method, folds, base, increment
            )
            for method in methods
        }

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_outer_train)
        x_test = scaler.transform(x[test_idx])

        fold_results = {}

        for method in methods:
            params = selected[method]

            result = evaluate(
                (
                    "alignment"
                    if method == "alignment"
                    else "composite"
                ),
                base,
                increment,
                params["c"],
                params["gamma"],
                params["weight"],
                x_train,
                y_outer_train,
                x_test,
                y[test_idx],
            )

            fold_results[method] = result

            rows.append({
                "dataset": name,
                "fold": fold,
                "method": method,
                "c": params["c"],
                "gamma": params["gamma"],
                "weight": result["weight"],
                "inner_auc": params["inner_auc"],
                "inner_balanced": params["inner_balanced"],
                "inner_ranking_loss": params[
                    "inner_ranking_loss"
                ],
                "auc": result["auc"],
                "accuracy": result["accuracy"],
                "balanced_accuracy": result[
                    "balanced_accuracy"
                ],
            })

        print(
            "Fold {:02d}/10 | Equal {:.4f} | "
            "Alignment {:.4f} | Grid {:.4f} | "
            "Continuous {:.4f}".format(
                fold,
                *[
                    fold_results[method]["auc"]
                    for method in methods
                ],
            ),
            flush=True,
        )

    return pd.DataFrame(rows)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.concat(
        [
            benchmark(name, x, y)
            for name, (x, y) in load_data().items()
        ],
        ignore_index=True,
    )

    summary = (
        results.groupby(["dataset", "method"])[
            ["auc", "accuracy", "balanced_accuracy"]
        ]
        .agg(["mean", "std"])
    )

    print("\n--- CONTINUOUS WEIGHT COMPARISON ---")
    print(summary.round(5).to_string())

    print("\n--- PAIRED AUC DELTAS ---")

    for dataset, group in results.groupby("dataset"):
        pivot = group.pivot(
            index="fold",
            columns="method",
            values="auc",
        )

        for reference in ("equal", "alignment", "grid"):
            delta = (
                pivot["continuous"] - pivot[reference]
            ).to_numpy()

            print(
                "{} | continuous vs {} | "
                "mean={:+.5f} | W/T/L={}".format(
                    dataset,
                    reference,
                    float(delta.mean()),
                    (
                        int((delta > 1e-12).sum()),
                        int((abs(delta) <= 1e-12).sum()),
                        int((delta < -1e-12).sum()),
                    ),
                )
            )

    print("\n--- LEARNED WEIGHTS ---")

    print(
        results.groupby(["dataset", "method"])["weight"]
        .agg(["mean", "std", "min", "max"])
        .round(5)
        .to_string()
    )

    results.to_csv(
        RESULTS_DIR / "continuous_mkl_folds.csv",
        index=False,
    )

    summary.to_csv(
        RESULTS_DIR / "continuous_mkl_summary.csv"
    )

    print("\nResults saved to {}".format(RESULTS_DIR))
    print(
        "Repeated outer folds overlap; "
        "comparisons are descriptive."
    )


if __name__ == "__main__":
    main()
