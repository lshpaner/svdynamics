
"""Nested-CV comparison of equal-weight and tuned additive MAGIC kernels.

Both models use the same baseline and increment RBF kernels. The adaptive
model selects the nonnegative increment multiplier on inner training folds.
This is kernel-weight tuning for KRR, not yet the SVDynamics SVM estimator.
"""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from magic_kernels import construct_kernels, split_features

DATA_PATH = Path(__file__).resolve().parent / "data" / "magic_sample.csv"
RIDGE_ALPHA = 0.1
WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
OUTER_FOLDS = 5
OUTER_REPEATS = 3
INNER_FOLDS = 3
SEED = 42


def kernel_scores(k_train, k_test, y_train, weight):
    train = k_train[0] + weight * k_train[1]
    test = k_test[0] + weight * k_test[1]
    a = train.copy()
    a.flat[::len(a) + 1] += RIDGE_ALPHA
    coefficients = cho_solve(
        cho_factor(a, check_finite=False),
        y_train,
        check_finite=False,
    )
    return test @ coefficients


def kernels_for_split(x_train, x_test):
    k = construct_kernels(*split_features(x_train, x_test))
    return (
        (k["K_base_train"], k["K_inc_train"]),
        (k["K_base_test"], k["K_inc_test"]),
    )


def evaluate(y, scores):
    return (
        accuracy_score(y, np.where(scores >= 0.0, 1, -1)),
        roc_auc_score(y, scores),
    )


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError("Run make research_magic_data first")

    frame = pd.read_csv(DATA_PATH)

    if "target" not in frame:
        raise ValueError("Missing target column")

    x = frame.drop(columns="target")
    y = frame["target"].to_numpy()

    if set(np.unique(y)) != {-1, 1}:
        raise ValueError("Expected -1/+1 target labels")

    outer = RepeatedStratifiedKFold(
        n_splits=OUTER_FOLDS,
        n_repeats=OUTER_REPEATS,
        random_state=SEED,
    )

    results = []

    for fold, (tr, te) in enumerate(outer.split(x, y), start=1):
        x_tr, x_te = x.iloc[tr], x.iloc[te]
        y_tr, y_te = y[tr], y[te]

        inner = StratifiedKFold(
            n_splits=INNER_FOLDS,
            shuffle=True,
            random_state=SEED + fold,
        )

        auc_by_weight = {weight: [] for weight in WEIGHTS}

        for it, iv in inner.split(x_tr, y_tr):
            inner_train = x_tr.iloc[it]
            inner_valid = x_tr.iloc[iv]

            k_train, k_valid = kernels_for_split(
                inner_train, inner_valid
            )

            for weight in WEIGHTS:
                scores = kernel_scores(
                    k_train, k_valid, y_tr[it], weight
                )
                auc_by_weight[weight].append(
                    roc_auc_score(y_tr[iv], scores)
                )

        # Prefer w=1 if inner-CV performance ties.
        chosen = max(
            WEIGHTS,
            key=lambda w: (
                np.mean(auc_by_weight[w]),
                -abs(w - 1.0),
            ),
        )

        k_train, k_test = kernels_for_split(x_tr, x_te)

        fixed = kernel_scores(k_train, k_test, y_tr, 1.0)
        tuned = kernel_scores(k_train, k_test, y_tr, chosen)

        fixed_acc, fixed_auc = evaluate(y_te, fixed)
        tuned_acc, tuned_auc = evaluate(y_te, tuned)

        results.append(
            dict(
                fold=fold,
                chosen_weight=chosen,
                fixed_accuracy=fixed_acc,
                tuned_accuracy=tuned_acc,
                delta_accuracy=tuned_acc - fixed_acc,
                fixed_auc=fixed_auc,
                tuned_auc=tuned_auc,
                delta_auc=tuned_auc - fixed_auc,
            )
        )

        print(
            "Fold {:02d}/{:02d} | weight={:<4g} | "
            "equal AUC={:.4f} | tuned AUC={:.4f}".format(
                fold,
                OUTER_FOLDS * OUTER_REPEATS,
                chosen,
                fixed_auc,
                tuned_auc,
            ),
            flush=True,
        )

    output = pd.DataFrame(results)

    print("\n--- MAGIC: Nested Adaptive Additive-Kernel KRR ---")
    print(
        "Outer: {} folds x {} repeats; inner: {} folds".format(
            OUTER_FOLDS, OUTER_REPEATS, INNER_FOLDS
        )
    )
    print("K = K_base + w*K_increment; fixed comparator w=1")
    print("Candidate weights: {}".format(WEIGHTS))

    print(
        output.drop(columns=["fold", "chosen_weight"])
        .agg(["mean", "std"])
        .round(5)
        .to_string()
    )

    print(
        "Selected weights: {}".format(
            dict(sorted(Counter(output["chosen_weight"]).items()))
        )
    )

    print(
        "Paired AUC wins/ties/losses: {}".format(
            (
                int((output.delta_auc > 1e-10).sum()),
                int((output.delta_auc.abs() <= 1e-10).sum()),
                int((output.delta_auc < -1e-10).sum()),
            )
        )
    )

    print("Repeated outer folds overlap; results are descriptive.")


if __name__ == "__main__":
    main()
