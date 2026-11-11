
import numpy as np
import pandas as pd

from sklearn.datasets import make_moons
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. Experiment configuration
# --------------------------------------------------
RANDOM_STATE = 42

N_SPLITS = 5
N_REPEATS = 10

RBF_GAMMA = 0.5
ETA = 0.1
LUSI_GAMMA = 1.0
TAU = 1.0


# --------------------------------------------------
# 2. LUSI implementation
# --------------------------------------------------
def fit_lusi(
    X,
    y,
    lusi_gamma=1.0,
    tau=1.0,
    eta=0.1,
    rbf_gamma=0.5
):
    """LUSI kernel solution from Liu et al. (2025), Eq. 1."""
    n = len(X)

    K = rbf_kernel(X, gamma=rbf_gamma)

    # Constant and linear statistical predicates
    Phi = np.column_stack([
        np.ones(n),
        X
    ])

    P = Phi @ Phi.T

    W = lusi_gamma * np.eye(n) + tau * P

    alpha = np.linalg.solve(
        W @ K + eta * np.eye(n),
        W @ y
    )

    return alpha


def classify(scores):
    """Convert prediction scores into {-1, 1} labels."""
    return np.where(scores >= 0, 1, -1)


# --------------------------------------------------
# 3. Generate dataset
# --------------------------------------------------
def main():

    X, y = make_moons(
        n_samples=100,
        noise=0.15,
        random_state=RANDOM_STATE
    )

    y = 2 * y - 1

    # --------------------------------------------------
    # 4. Repeated stratified cross-validation
    # --------------------------------------------------
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE
    )

    results = []

    for split_idx, (train_idx, test_idx) in enumerate(
        cv.split(X, y)
    ):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        # Fit scaler only on the training fold
        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # --------------------------------------------------
        # 5. Fit LUSI
        # --------------------------------------------------
        alpha = fit_lusi(
            X_train,
            y_train,
            lusi_gamma=LUSI_GAMMA,
            tau=TAU,
            eta=ETA,
            rbf_gamma=RBF_GAMMA
        )

        K_test = rbf_kernel(
            X_test,
            X_train,
            gamma=RBF_GAMMA
        )

        lusi_scores = K_test @ alpha
        lusi_pred = classify(lusi_scores)

        # --------------------------------------------------
        # 6. Fit KRR
        # --------------------------------------------------
        krr = KernelRidge(
            alpha=ETA,
            kernel="rbf",
            gamma=RBF_GAMMA
        )

        krr.fit(X_train, y_train)

        krr_scores = krr.predict(X_test)
        krr_pred = classify(krr_scores)

        # --------------------------------------------------
        # 7. Evaluate both models
        # --------------------------------------------------
        lusi_accuracy = accuracy_score(
            y_test,
            lusi_pred
        )

        krr_accuracy = accuracy_score(
            y_test,
            krr_pred
        )

        disagreements = np.sum(
            lusi_pred != krr_pred
        )

        mean_score_difference = np.mean(
            np.abs(lusi_scores - krr_scores)
        )

        results.append({
            "repeat": split_idx // N_SPLITS + 1,
            "fold": split_idx % N_SPLITS + 1,
            "lusi_accuracy": lusi_accuracy,
            "krr_accuracy": krr_accuracy,
            "accuracy_difference": (
                lusi_accuracy - krr_accuracy
            ),
            "prediction_disagreements": disagreements,
            "mean_score_difference": mean_score_difference
        })

    # --------------------------------------------------
    # 8. Aggregate results
    # --------------------------------------------------
    df_results = pd.DataFrame(results)

    print("\n--- Experiment Configuration ---")
    print("Dataset: make_moons")
    print("Observations:", len(X))
    print("Folds:", N_SPLITS)
    print("Repeats:", N_REPEATS)
    print("Total evaluations:", len(df_results))

    print("\n--- Cross-Validation Accuracy ---")

    summary = df_results[
        [
            "lusi_accuracy",
            "krr_accuracy",
            "accuracy_difference"
        ]
    ].agg(["mean", "std", "min", "max"])

    print(summary)

    # --------------------------------------------------
    # 9. Paired comparison diagnostics
    # --------------------------------------------------
    print("\n--- Paired Model Comparison ---")

    print(
        "LUSI wins:",
        (df_results["accuracy_difference"] > 0).sum()
    )

    print(
        "KRR wins:",
        (df_results["accuracy_difference"] < 0).sum()
    )

    print(
        "Ties:",
        (df_results["accuracy_difference"] == 0).sum()
    )

    print(
        "Total prediction disagreements:",
        df_results["prediction_disagreements"].sum()
    )

    print(
        "Mean absolute score difference:",
        df_results["mean_score_difference"].mean()
    )

    # --------------------------------------------------
    # 10. Save results
    # --------------------------------------------------
    output_file = "lusi_cv_results.csv"

    df_results.to_csv(
        output_file,
        index=False
    )

    print("\nResults saved to:", output_file)


if __name__ == "__main__":
    main()
