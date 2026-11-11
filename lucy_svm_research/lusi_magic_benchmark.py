"""Compare conventional KRR and invariant-weighted LUSI on MAGIC.

Independent experimental configuration, not an exact reproduction of
Liu et al. (2025), Section 6.3.
Run from the repository root: make research_magic_lusi
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from magic_kernels import construct_kernels, split_features

RANDOM_STATE = 42
TEST_SIZE = 0.20
RIDGE_ALPHA = 0.1
LUSI_GAMMA = 1.0
LUSI_TAU = 1.0
LUSI_ETA = 0.1
DATA_PATH = Path(__file__).resolve().parent / "data" / "magic_sample.csv"


def metrics(y_true, scores):
    predictions = np.where(scores >= 0.0, 1, -1)
    return accuracy_score(y_true, predictions), roc_auc_score(y_true, scores), predictions


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError("Missing MAGIC sample; run make research_magic_data")
    df = pd.read_csv(DATA_PATH)
    if "target" not in df:
        raise ValueError("Missing target column")
    X = df.drop(columns="target")
    y = df["target"].to_numpy()
    if set(np.unique(y)) != {-1, 1}:
        raise ValueError("Targets must be -1 and +1")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_base_train, X_base_test, X_inc_train, X_inc_test = split_features(
        X_train, X_test
    )
    kernels = construct_kernels(
        X_base_train, X_base_test, X_inc_train, X_inc_test
    )
    K_train = kernels["K_base_train"]
    K_test = kernels["K_base_test"]
    n = len(y_train)
    eye = np.eye(n)

    # Conventional KRR baseline.
    alpha_krr = np.linalg.solve(K_train + RIDGE_ALPHA * eye, y_train)
    scores_krr = K_test @ alpha_krr
    reference = KernelRidge(alpha=RIDGE_ALPHA, kernel="precomputed")
    reference.fit(K_train, y_train)
    max_difference = np.max(np.abs(scores_krr - reference.predict(K_test)))
    if not np.allclose(scores_krr, reference.predict(K_test), atol=1e-8):
        raise AssertionError("KRR does not match sklearn")

    # Invariant features: intercept plus the eight held-out feature columns.
    # Scaling is already fitted on training data by split_features().
    phi = np.column_stack((np.ones(n), X_inc_train))
    P = phi @ phi.T
    W = LUSI_GAMMA * eye + LUSI_TAU * P

    # Invariant-weighted kernel ridge solution:
    # (W K + eta I) alpha = W y.
    # This is not a hinge-loss support vector machine.
    alpha_lusi = np.linalg.solve(W @ K_train + LUSI_ETA * eye, W @ y_train)
    residual = np.linalg.norm(
        (W @ K_train + LUSI_ETA * eye) @ alpha_lusi - W @ y_train
    ) / max(np.linalg.norm(W @ y_train), 1e-12)
    if not np.isfinite(residual) or residual > 1e-6:
        raise AssertionError("LUSI linear system residual too large: {}".format(residual))
    scores_lusi = K_test @ alpha_lusi

    acc_krr, auc_krr, pred_krr = metrics(y_test, scores_krr)
    acc_lusi, auc_lusi, pred_lusi = metrics(y_test, scores_lusi)

    print("\n--- MAGIC: KRR vs Invariant-Weighted LUSI ---")
    print("Train/test: {}/{}".format(n, len(y_test)))
    print("Baseline features: fLength, fWidth")
    print("Invariant features: 8 standardized additional features + intercept")
    print("RBF gamma: 0.5")
    print("KRR alpha: {}".format(RIDGE_ALPHA))
    print("LUSI gamma/tau/eta: {}/{}/{}".format(
        LUSI_GAMMA, LUSI_TAU, LUSI_ETA
    ))
    print("KRR accuracy: {:.4f}; ROC-AUC: {:.4f}".format(acc_krr, auc_krr))
    print("LUSI accuracy: {:.4f}; ROC-AUC: {:.4f}".format(acc_lusi, auc_lusi))
    print("Accuracy difference (LUSI - KRR): {:+.4f}".format(acc_lusi - acc_krr))
    print("ROC-AUC difference (LUSI - KRR): {:+.4f}".format(auc_lusi - auc_krr))
    print("Changed test predictions: {} / {}".format(
        np.count_nonzero(pred_krr != pred_lusi), len(y_test)
    ))
    print("Mean absolute score difference: {:.6f}".format(
        np.mean(np.abs(scores_lusi - scores_krr))
    ))
    print("LUSI relative solve residual: {:.3e}".format(residual))
    print("KRR max difference vs sklearn: {:.3e}".format(max_difference))


if __name__ == "__main__":
    main()
