"""MAGIC KRR vs LUSI tau sensitivity on a fixed stratified split.

Independent experiment, not an exact reproduction of Liu et al. (2025).
Run from repository root: make research_magic_tau
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from magic_kernels import construct_kernels, split_features

DATA_PATH = Path(__file__).resolve().parent / "data" / "magic_sample.csv"
TAUS = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0)
GAMMA = 1.0
ETA = 0.1
SEED = 42


def evaluate(y, scores):
    predictions = np.where(scores >= 0, 1, -1)
    return accuracy_score(y, predictions), roc_auc_score(y, scores), predictions


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError("Run make research_magic_data first")
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns="target")
    y = df["target"].to_numpy()
    if set(np.unique(y)) != {-1, 1}:
        raise ValueError("Expected -1/+1 target encoding")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    features = split_features(X_train, X_test)
    kernels = construct_kernels(*features)
    K = kernels["K_base_train"]
    K_test = kernels["K_base_test"]
    phi = np.column_stack((np.ones(len(y_train)), features[2]))
    n = len(y_train)
    eye = np.eye(n)

    # Equivalent symmetric system: (K + eta * W^{-1}) alpha = y,
    # where W = gamma I + tau Phi Phi^T. Woodbury avoids a dense inverse.
    # W^{-1} = I/gamma - (tau/gamma^2) Phi
    #           (I + (tau/gamma) Phi^T Phi)^{-1} Phi^T.
    gram_phi = phi.T @ phi
    rows = []
    reference_scores = None
    reference_predictions = None
    for tau in TAUS:
        small = np.eye(phi.shape[1]) + (tau / GAMMA) * gram_phi
        correction = np.linalg.solve(small, phi.T)
        w_inv = (eye / GAMMA) - (tau / (GAMMA ** 2)) * (phi @ correction)
        system = K + ETA * w_inv
        system = (system + system.T) / 2.0
        factor = cho_factor(system, lower=True, check_finite=True)
        alpha = cho_solve(factor, y_train)
        residual = np.linalg.norm(system @ alpha - y_train) / np.linalg.norm(y_train)
        scores = K_test @ alpha
        accuracy, auc, predictions = evaluate(y_test, scores)
        if tau == 0:
            reference_scores = scores
            reference_predictions = predictions
            baseline_alpha = np.linalg.solve(K + (ETA / GAMMA) * eye, y_train)
            baseline_difference = np.max(np.abs(K_test @ baseline_alpha - scores))
            if baseline_difference > 1e-7:
                raise AssertionError("tau=0 does not recover conventional KRR")
        rows.append({
            "tau": tau,
            "accuracy": accuracy,
            "roc_auc": auc,
            "changed_predictions": int(np.count_nonzero(predictions != reference_predictions)),
            "mean_abs_score_difference": float(np.mean(np.abs(scores - reference_scores))),
            "relative_residual": float(residual),
        })
    results = pd.DataFrame(rows)
    print("\n--- MAGIC: LUSI Tau Sensitivity ---")
    print("Train/test: {}/{}; gamma={}; eta={}".format(n, len(y_test), GAMMA, ETA))
    print("Baseline features: fLength, fWidth; invariant features: 8 + intercept")
    print("tau=0 vs KRR maximum score difference: {:.3e}".format(baseline_difference))
    print(results.to_string(index=False, float_format=lambda v: "{:.6f}".format(v)))
    print("\nDescriptive only: same held-out test set for all tau values.")
    print("Do not select tau by maximizing this test-set performance.")


if __name__ == "__main__":
    main()
