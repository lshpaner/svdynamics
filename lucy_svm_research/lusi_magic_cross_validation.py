"""Nested repeated CV of MAGIC KRR vs invariant-weighted LUSI.

Independent experimental configuration, not an exact replication of Liu et al.
Run from repository root: make research_magic_cv
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from magic_kernels import construct_kernels, split_features

DATA_PATH = Path(__file__).resolve().parent / "data" / "magic_sample.csv"
TAUS = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0)
RIDGE_ALPHA = 0.1
GAMMA = 1.0
ETA = 0.1
SEED = 42


def scores_for_tau(k_train, k_test, phi, y_train, tau):
    """Solve (K + eta W^-1) a = y, W = gamma I + tau Phi Phi^T."""
    n = len(y_train)
    identity = np.eye(n)
    if tau == 0:
        system = k_train + (ETA / GAMMA) * identity
    else:
        small = np.eye(phi.shape[1]) + (tau / GAMMA) * (phi.T @ phi)
        correction = np.linalg.solve(small, phi.T)
        w_inv = identity / GAMMA - (tau / GAMMA**2) * (phi @ correction)
        system = k_train + ETA * w_inv
    system = (system + system.T) / 2
    alpha = cho_solve(cho_factor(system, lower=True, check_finite=False), y_train,
                      check_finite=False)
    return k_test @ alpha


def score_pair(y_true, scores):
    return (accuracy_score(y_true, np.where(scores >= 0, 1, -1)),
            roc_auc_score(y_true, scores))


def prepare(x_train, x_test):
    features = split_features(x_train, x_test)
    kernels = construct_kernels(*features)
    phi = np.column_stack((np.ones(len(x_train)), features[2]))
    return kernels["K_base_train"], kernels["K_base_test"], phi


def select_tau(x_train, y_train, inner_splits, seed):
    splitter = StratifiedKFold(n_splits=inner_splits, shuffle=True,
                               random_state=seed)
    fold_scores = {tau: [] for tau in TAUS}
    for train_idx, valid_idx in splitter.split(x_train, y_train):
        x_inner = x_train.iloc[train_idx]
        x_valid = x_train.iloc[valid_idx]
        y_inner = y_train[train_idx]
        y_valid = y_train[valid_idx]
        k_train, k_valid, phi = prepare(x_inner, x_valid)
        for tau in TAUS:
            predictions = scores_for_tau(k_train, k_valid, phi, y_inner, tau)
            fold_scores[tau].append(roc_auc_score(y_valid, predictions))
    # Fixed deterministic tie break: prefer the smaller tau.
    return max(TAUS, key=lambda tau: (np.mean(fold_scores[tau]), -tau))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    args = parser.parse_args()
    if min(args.repeats, args.outer_folds, args.inner_folds) < 1 or min(
            args.outer_folds, args.inner_folds) < 2:
        parser.error("repeats must be >=1 and folds must be >=2")
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run make research_magic_data first")
    df = pd.read_csv(DATA_PATH)
    x = df.drop(columns="target")
    y = df["target"].to_numpy()
    if set(np.unique(y)) != {-1, 1}:
        raise ValueError("Expected target labels -1 and +1")

    outer = RepeatedStratifiedKFold(n_splits=args.outer_folds,
                                    n_repeats=args.repeats,
                                    random_state=SEED)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(outer.split(x, y), start=1):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        tau = select_tau(x_train, y_train, args.inner_folds, SEED + fold)
        k_train, k_test, phi = prepare(x_train, x_test)
        krr_scores = scores_for_tau(k_train, k_test, phi, y_train, 0.0)
        lusi_scores = scores_for_tau(k_train, k_test, phi, y_train, tau)
        krr_acc, krr_auc = score_pair(y_test, krr_scores)
        lusi_acc, lusi_auc = score_pair(y_test, lusi_scores)
        rows.append({"fold": fold, "tau": tau, "krr_accuracy": krr_acc,
                     "lusi_accuracy": lusi_acc, "krr_roc_auc": krr_auc,
                     "lusi_roc_auc": lusi_auc,
                     "delta_accuracy": lusi_acc - krr_acc,
                     "delta_roc_auc": lusi_auc - krr_auc})
        print("Fold {:02d}/{:02d} | tau={:<6g} | KRR AUC={:.4f} | LUSI AUC={:.4f}".format(
            fold, args.outer_folds * args.repeats, tau, krr_auc, lusi_auc), flush=True)

    results = pd.DataFrame(rows)
    cols = ["krr_accuracy", "lusi_accuracy", "delta_accuracy",
            "krr_roc_auc", "lusi_roc_auc", "delta_roc_auc"]
    print("\n--- MAGIC: Nested Repeated Stratified CV ---")
    print("Outer: {} folds x {} repeats = {} paired evaluations".format(
        args.outer_folds, args.repeats, len(results)))
    print("Inner: {} stratified folds; select tau by mean ROC-AUC".format(args.inner_folds))
    print("Fixed KRR alpha={}, RBF gamma=0.5, LUSI gamma={}, eta={}".format(
        RIDGE_ALPHA, GAMMA, ETA))
    print(results[cols].agg(["mean", "std"]).to_string(float_format=lambda v: "{:.5f}".format(v)))
    print("Selected taus:", dict(sorted(Counter(results["tau"]).items())))
    print("Paired accuracy wins/ties/losses:",
          (int((results.delta_accuracy > 0).sum()),
           int((results.delta_accuracy == 0).sum()),
           int((results.delta_accuracy < 0).sum())))
    print("Note: outer folds overlap across repeats; differences are descriptive, not independent trials.")


if __name__ == "__main__":
    main()
