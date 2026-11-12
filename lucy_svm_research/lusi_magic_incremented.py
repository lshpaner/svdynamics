
"""Compare baseline KRR with unrestricted incremented-kernel KRR on MAGIC.

Independent experimental configuration inspired by Liu et al. (2025),
Section 6.3. This does not reproduce the paper's feature-selection
and tuning protocol.

Run from repository root: make research_magic_incremented
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from magic_kernels import construct_kernels, split_features


DATA_PATH = Path(__file__).resolve().parent / "data" / "magic_sample.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20
RIDGE_ALPHA = 0.1


def evaluate(y_true, scores):
    return (
        accuracy_score(y_true, np.where(scores >= 0, 1, -1)),
        roc_auc_score(y_true, scores),
    )


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            "MAGIC sample missing; run make research_magic_data"
        )

    df = pd.read_csv(DATA_PATH)

    if "target" not in df.columns:
        raise ValueError("Expected a target column")

    X = df.drop(columns="target")
    y = df["target"].to_numpy()

    if set(np.unique(y)) != {-1, 1}:
        raise ValueError("Expected targets encoded as -1 and +1")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    feature_arrays = split_features(X_train, X_test)
    kernels = construct_kernels(*feature_arrays)

    K_base_train = kernels["K_base_train"]
    K_base_test = kernels["K_base_test"]
    K_inc_train = kernels["K_inc_train"]
    K_inc_test = kernels["K_inc_test"]

    # Unrestricted positive additive increment, not LUSIAdj.
    K_full_train = K_base_train + K_inc_train
    K_full_test = K_base_test + K_inc_test

    for name, matrix in (
        ("base", K_base_train),
        ("incremented", K_full_train),
    ):
        if not np.allclose(matrix, matrix.T, atol=1e-10):
            raise AssertionError(
                "{} training kernel is not symmetric".format(name)
            )

        if not np.isfinite(matrix).all():
            raise AssertionError(
                "{} training kernel contains nonfinite values".format(name)
            )

    base = KernelRidge(alpha=RIDGE_ALPHA, kernel="precomputed")
    full = KernelRidge(alpha=RIDGE_ALPHA, kernel="precomputed")

    base.fit(K_base_train, y_train)
    full.fit(K_full_train, y_train)

    scores_base = base.predict(K_base_test)
    scores_full = full.predict(K_full_test)

    base_acc, base_auc = evaluate(y_test, scores_base)
    full_acc, full_auc = evaluate(y_test, scores_full)

    changed = np.count_nonzero(
        (scores_base >= 0) != (scores_full >= 0)
    )

    print("\n--- MAGIC: Unrestricted Incremented-Kernel KRR ---")
    print("Train/test: {}/{}".format(len(y_train), len(y_test)))
    print("Baseline features: fLength, fWidth")
    print("Increment features: remaining 8 standardized features")
    print("K_full = K_base + K_increment (equal weights)")
    print(
        "RBF gamma (both kernels): 0.5; ridge alpha: {}".format(
            RIDGE_ALPHA
        )
    )

    print(
        "Baseline KRR accuracy: {:.4f}; ROC-AUC: {:.4f}".format(
            base_acc, base_auc
        )
    )
    print(
        "Incremented KRR accuracy: {:.4f}; ROC-AUC: {:.4f}".format(
            full_acc, full_auc
        )
    )
    print(
        "Accuracy difference: {:+.4f}".format(
            full_acc - base_acc
        )
    )
    print(
        "ROC-AUC difference: {:+.4f}".format(
            full_auc - base_auc
        )
    )
    print(
        "Changed test predictions: {} / {}".format(
            changed, len(y_test)
        )
    )
    print(
        "Note: one held-out split, no hyperparameter optimization."
    )


if __name__ == "__main__":
    main()
