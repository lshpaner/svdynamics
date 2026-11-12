
"""Final matched SVM kernel benchmark on Moons and MAGIC.

Methods:
    rbf:       conventional RBF SVM
    equal:     equal-weight additive composite
    alignment: kernel-target alignment
    grid:      inner-CV ROC-AUC weight selection
    dual_mkl:  minimize optimized SVM dual over kernel weights

Fixed C and gamma for every method.
Outer folds are identical for all methods.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import minimize_scalar
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
from sklearn.svm import SVC


SEED = 42
C = 10.0
GAMMA = 0.5
OUTER_SPLITS = 5
OUTER_REPEATS = 2
INNER_SPLITS = 3

WEIGHT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
METHODS = ("rbf", "equal", "alignment", "grid", "dual_mkl")

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MAGIC_PATH = ROOT / "data" / "magic_sample.csv"


def load_datasets():
    x, y = make_moons(
        n_samples=600,
        noise=0.25,
        random_state=SEED,
    )
    datasets = {"Moons": (x, y)}

    if MAGIC_PATH.exists():
        frame = pd.read_csv(MAGIC_PATH)
        candidates = (
            "target", "class", "label", "Class", "Target"
        )
        target_col = next(
            (c for c in candidates if c in frame.columns),
            frame.columns[-1],
        )

        raw = frame[target_col]

        if raw.isna().any():
            raise ValueError("Missing MAGIC labels")

        values = set(raw.unique())

        if values == {-1, 1}:
            y = (raw.to_numpy() == 1).astype(int)
        elif values == {0, 1}:
            y = raw.to_numpy().astype(int)
        elif values == {"g", "h"}:
            y = (raw.to_numpy() == "g").astype(int)
        else:
            raise ValueError(
                "Unexpected MAGIC labels: {}".format(values)
            )

        features = (
            frame.drop(columns=[target_col])
            .select_dtypes(include=[np.number])
        )

        if features.shape[1] != 10:
            raise ValueError("MAGIC must have 10 predictors")

        if features.isna().any().any():
            raise ValueError("Missing MAGIC predictors")

        datasets["MAGIC"] = (
            features.to_numpy(dtype=float),
            y,
        )
    else:
        print("MAGIC CSV not found, skipping MAGIC.")

    return datasets


def component_matrices(x, z=None):
    """Compute two feature-group RBF matrices."""
    if z is None:
        z = x

    midpoint = x.shape[1] // 2
    groups = (
        np.arange(midpoint),
        np.arange(midpoint, x.shape[1]),
    )

    return [
        rbf_kernel(
            x[:, columns],
            z[:, columns],
            gamma=GAMMA,
        )
        for columns in groups
    ]


def combine(kernels, weight):
    return (
        float(weight) * kernels[0]
        + (1.0 - float(weight)) * kernels[1]
    )


def fit_svm(kernel, y):
    model = SVC(
        C=C,
        kernel="precomputed",
    )
    model.fit(kernel, y)
    return model


def dual_value(model, kernel):
    signed = np.zeros(kernel.shape[0], dtype=float)
    signed[model.support_] = model.dual_coef_[0]

    return float(
        np.abs(signed).sum()
        - 0.5 * signed @ kernel @ signed
    )


def learn_dual_weight(kernels, y):
    """Minimize maximized SVM dual objective over the simplex."""
    cache = {}

    def objective(weight):
        weight = float(weight)
        key = round(weight, 12)

        if key not in cache:
            gram = combine(kernels, weight)
            model = fit_svm(gram, y)
            cache[key] = dual_value(model, gram)

        return cache[key]

    result = minimize_scalar(
        objective,
        bounds=(0.0, 1.0),
        method="bounded",
        options={
            "xatol": 1e-4,
            "maxiter": 100,
        },
    )

    if not result.success:
        raise RuntimeError("Dual MKL optimizer did not converge")

    candidates = (0.0, float(result.x), 1.0)

    best = min(
        candidates,
        key=objective,
    )

    return float(best), float(objective(best))


def alignment_weight(kernels, y):
    """Centered kernel-target alignment, with nonnegative weights."""
    labels = 2.0 * np.asarray(y, dtype=float) - 1.0
    target = np.outer(labels, labels)
    target -= target.mean(axis=0, keepdims=True)
    target -= target.mean(axis=1, keepdims=True)
    target += target.mean()

    scores = []

    for kernel in kernels:
        centered = (
            kernel
            - kernel.mean(axis=0, keepdims=True)
            - kernel.mean(axis=1, keepdims=True)
            + kernel.mean()
        )

        denominator = (
            np.linalg.norm(centered)
            * np.linalg.norm(target)
        )

        score = (
            np.sum(centered * target) / denominator
            if denominator > 0 else 0.0
        )
        scores.append(max(0.0, float(score)))

    total = sum(scores)

    return scores[0] / total if total > 0 else 0.5


def learn_grid_weight(x, y, outer_fold):
    splitter = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=SEED + outer_fold,
    )

    scores = {w: [] for w in WEIGHT_GRID}

    for train_idx, valid_idx in splitter.split(x, y):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_valid = scaler.transform(x[valid_idx])

        train_kernels = component_matrices(x_train)
        valid_kernels = component_matrices(
            x_valid, x_train
        )

        for weight in WEIGHT_GRID:
            model = fit_svm(
                combine(train_kernels, weight),
                y[train_idx],
            )
            decision = model.decision_function(
                combine(valid_kernels, weight)
            )
            scores[weight].append(
                roc_auc_score(y[valid_idx], decision)
            )

    # Prefer weights near 0.5 when AUC is tied.
    return max(
        WEIGHT_GRID,
        key=lambda w: (
            np.mean(scores[w]),
            -abs(w - 0.5),
        ),
    )


def evaluate_model(
    method, train_kernels, test_kernels,
    full_train_kernel, full_test_kernel,
    y_train, y_test, weight
):
    if method == "rbf":
        train_gram = full_train_kernel
        test_gram = full_test_kernel
    else:
        train_gram = combine(train_kernels, weight)
        test_gram = combine(test_kernels, weight)

    model = fit_svm(train_gram, y_train)

    predictions = model.predict(test_gram)
    decision = model.decision_function(test_gram)

    return {
        "auc": roc_auc_score(y_test, decision),
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
        "dual_objective": dual_value(
            model, train_gram
        ),
    }


def benchmark(name, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)

    splitter = RepeatedStratifiedKFold(
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
        splitter.split(x, y),
        start=1,
    ):
        scaler = StandardScaler()

        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])

        y_train = y[train_idx]
        y_test = y[test_idx]

        train_kernels = component_matrices(x_train)
        test_kernels = component_matrices(
            x_test, x_train
        )

        full_train = rbf_kernel(
            x_train, x_train, gamma=GAMMA
        )
        full_test = rbf_kernel(
            x_test, x_train, gamma=GAMMA
        )

        dual_weight, dual_optimum = learn_dual_weight(
            train_kernels, y_train
        )

        weights = {
            "rbf": np.nan,
            "equal": 0.5,
            "alignment": alignment_weight(
                train_kernels, y_train
            ),
            "grid": learn_grid_weight(
                x[train_idx], y_train, fold
            ),
            "dual_mkl": dual_weight,
        }

        fold_auc = {}

        for method in METHODS:
            result = evaluate_model(
                method,
                train_kernels,
                test_kernels,
                full_train,
                full_test,
                y_train,
                y_test,
                weights[method],
            )

            fold_auc[method] = result["auc"]

            rows.append({
                "dataset": name,
                "fold": fold,
                "method": method,
                "weight": weights[method],
                "mkl_optimum": (
                    dual_optimum
                    if method == "dual_mkl"
                    else np.nan
                ),
                **result,
            })

        print(
            "Fold {:02d}/10 | "
            "RBF {:.4f} | Equal {:.4f} | "
            "Align {:.4f} | Grid {:.4f} | "
            "MKL {:.4f} | w={:.4f}".format(
                fold,
                fold_auc["rbf"],
                fold_auc["equal"],
                fold_auc["alignment"],
                fold_auc["grid"],
                fold_auc["dual_mkl"],
                dual_weight,
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

    summary = (
        results.groupby(["dataset", "method"])[
            ["auc", "accuracy", "balanced_accuracy"]
        ]
        .agg(["mean", "std"])
    )

    print("\n--- FINAL MKL BENCHMARK ---")
    print(summary.round(5).to_string())

    print("\n--- PAIRED MKL ROC-AUC COMPARISONS ---")

    for dataset, group in results.groupby("dataset"):
        pivot = group.pivot(
            index="fold",
            columns="method",
            values="auc",
        )

        for reference in (
            "rbf", "equal", "alignment", "grid"
        ):
            delta = (
                pivot["dual_mkl"] - pivot[reference]
            ).to_numpy()

            print(
                "{} | MKL vs {} | Delta={:+.5f} "
                "| W/T/L={}".format(
                    dataset,
                    reference,
                    delta.mean(),
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
        .round(5).to_string()
    )

    results.to_csv(
        RESULTS / "final_mkl_folds.csv",
        index=False,
    )
    summary.to_csv(
        RESULTS / "final_mkl_summary.csv"
    )

    print("\nResults saved to {}".format(RESULTS))
    print("Repeated outer folds overlap; results are descriptive.")


if __name__ == "__main__":
    main()
