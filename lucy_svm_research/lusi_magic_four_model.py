
"""Four-model MAGIC comparison with a representable LUSI kernel increment.

Models: baseline KRR, LUSI, LUSIAdj, unrestricted incremented KRR.
This is an independent Section 6.3-style experiment, not an exact
replication of the paper's numerical setup.

Run: make research_magic_four_model
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import eigsh
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from magic_kernels import construct_kernels, split_features

DATA_PATH = Path(__file__).resolve().parent / "data" / "magic_sample.csv"
RIDGE_ALPHA = 0.1
GAMMA = 1.0
ETA = 0.1
CONSTRAINT_FRACTION = 0.5


def scores_from_kernel(k_train, k_test, y, alpha):
    n = len(y)
    regularized = k_train + alpha * np.eye(n)
    coefficients = cho_solve(cho_factor(regularized, check_finite=True), y)
    return k_test @ coefficients, coefficients


def metrics(y, scores):
    return (accuracy_score(y, np.where(scores >= 0, 1, -1)),
            roc_auc_score(y, scores))


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError("Run make research_magic_data first")
    frame = pd.read_csv(DATA_PATH)
    if "target" not in frame.columns:
        raise ValueError("Expected target column")
    x = frame.drop(columns="target")
    y = frame["target"].to_numpy(dtype=float)
    if set(np.unique(y)) != {-1.0, 1.0}:
        raise ValueError("Expected binary labels -1 and +1")

    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )
    kernels = construct_kernels(*split_features(x_tr, x_te))
    base_tr = kernels["K_base_train"]
    base_te = kernels["K_base_test"]
    inc_tr = kernels["K_inc_train"]
    inc_te = kernels["K_inc_test"]

    # Constrained decrement: D = c * K_inc.
    # Choose c using training data only so the decrement eigenvalues
    # are strictly below eta/gamma.
    max_eigenvalue = float(
        eigsh(inc_tr, k=1, which="LA", return_eigenvectors=False)[0]
    )
    if max_eigenvalue <= 0:
        raise ValueError("Increment kernel must have positive largest eigenvalue")
    c = CONSTRAINT_FRACTION * (ETA / GAMMA) / max_eigenvalue
    decrement_tr = c * inc_tr
    decrement_te = c * inc_te

    # The implied predicate eigenvalues are
    # lambda = gamma^2*d/(eta-gamma*d).
    # These are nonnegative because d < eta/gamma.
    adjusted_tr = base_tr - decrement_tr
    adjusted_te = base_te - decrement_te
    regularization = ETA / GAMMA

    baseline_scores, _ = scores_from_kernel(
        base_tr, base_te, y_tr, regularization
    )

    # LUSI fits using K' but predicts using original K.
    _, lusi_coefficients = scores_from_kernel(
        adjusted_tr, adjusted_te, y_tr, regularization
    )
    lusi_scores = base_te @ lusi_coefficients

    # LUSIAdj uses the adjusted prediction kernel.
    lusi_adj_scores = adjusted_te @ lusi_coefficients

    # Unrestricted additive kernel.
    full_scores, _ = scores_from_kernel(
        base_tr + inc_tr, base_te + inc_te, y_tr, regularization
    )

    results = {
        "Baseline KRR": baseline_scores,
        "LUSI": lusi_scores,
        "LUSIAdj": lusi_adj_scores,
        "Unrestricted incremented KRR": full_scores,
    }

    print("\n--- MAGIC: Four-Model Comparison ---")
    print("Train/test: {}/{}".format(len(y_tr), len(y_te)))
    print("RBF gamma: 0.5; ridge alpha: {:.3f}".format(regularization))
    print("LUSI gamma: {}; eta: {}".format(GAMMA, ETA))
    print("Max eigenvalue K_inc: {:.6f}".format(max_eigenvalue))
    print("Constrained decrement multiplier c: {:.9f}".format(c))
    print(
        "Max decrement eigenvalue: {:.6f} (< {:.6f})".format(
            c * max_eigenvalue, ETA / GAMMA
        )
    )
    print("{:<29} {:>10} {:>10}".format("Model", "Accuracy", "ROC-AUC"))
    for name, values in results.items():
        accuracy, auc = metrics(y_te, values)
        print(
            "{:<29} {:>10.4f} {:>10.4f}".format(
                name, accuracy, auc
            )
        )
    print("Note: single held-out split; no tuning or significance claim.")


if __name__ == "__main__":
    main()
