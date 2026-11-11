
import numpy as np

from sklearn.datasets import make_moons
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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

    print("Kernel symmetric:", np.allclose(K, K.T))
    print("Minimum eigenvalue:", np.linalg.eigvalsh(K).min())

    # Simple statistical predicates: constant and feature moments
    Phi = np.column_stack([np.ones(n), X])
    P = Phi @ Phi.T

    W = lusi_gamma * np.eye(n) + tau * P

    alpha = np.linalg.solve(
        W @ K + eta * np.eye(n),
        W @ y
    )

    return alpha


if __name__ == "__main__":

    # --------------------------------------------------
    # 1. Generate synthetic classification data
    # --------------------------------------------------
    X, y = make_moons(
        n_samples=100,
        noise=0.15,
        random_state=42
    )

    # Convert labels from {0, 1} to {-1, 1}
    y = 2 * y - 1

    # --------------------------------------------------
    # 2. Train/test split
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42
    )

    print(
        "Training labels:",
        np.unique(y_train, return_counts=True)
    )

    print(
        "Testing labels:",
        np.unique(y_test, return_counts=True)
    )

    print("Training observations:", len(X_train))
    print("Testing observations:", len(X_test))

    # --------------------------------------------------
    # 3. Standardize features without data leakage
    # --------------------------------------------------
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --------------------------------------------------
    # 4. Fit LUSI with statistical predicates
    # --------------------------------------------------
    alpha = fit_lusi(
        X_train,
        y_train,
        lusi_gamma=1.0,
        tau=1.0,
        eta=0.1,
        rbf_gamma=0.5
    )

    # --------------------------------------------------
    # 5. Fit Kernel Ridge Regression baseline
    # --------------------------------------------------
    krr = KernelRidge(
        alpha=0.1,
        kernel="rbf",
        gamma=0.5
    )

    krr.fit(X_train, y_train)

    # --------------------------------------------------
    # 6. Training predictions
    # --------------------------------------------------
    K_train = rbf_kernel(
        X_train,
        X_train,
        gamma=0.5
    )

    lusi_train_scores = K_train @ alpha

    lusi_train_pred = np.where(
        lusi_train_scores >= 0,
        1,
        -1
    )

    krr_train_scores = krr.predict(X_train)

    krr_train_pred = np.where(
        krr_train_scores >= 0,
        1,
        -1
    )

    print("\n--- Training Results ---")

    print(
        "LUSI train accuracy:",
        accuracy_score(y_train, lusi_train_pred)
    )

    print(
        "KRR train accuracy:",
        accuracy_score(y_train, krr_train_pred)
    )

    # --------------------------------------------------
    # 7. Held-out test predictions
    # --------------------------------------------------
    K_test = rbf_kernel(
        X_test,
        X_train,
        gamma=0.5
    )

    lusi_scores = K_test @ alpha
    krr_scores = krr.predict(X_test)

    lusi_test_pred = np.where(
        lusi_scores >= 0,
        1,
        -1
    )

    krr_test_pred = np.where(
        krr_scores >= 0,
        1,
        -1
    )

    print("\n--- Test Results ---")

    print(
        "LUSI test accuracy:",
        accuracy_score(y_test, lusi_test_pred)
    )

    print(
        "KRR test accuracy:",
        accuracy_score(y_test, krr_test_pred)
    )

    print(
        "Test prediction disagreements:",
        np.sum(lusi_test_pred != krr_test_pred)
    )

    print(
        "Maximum score difference:",
        np.max(np.abs(lusi_scores - krr_scores))
    )

    print(
        "Mean score difference:",
        np.mean(np.abs(lusi_scores - krr_scores))
    )

    # --------------------------------------------------
    # 8. LUSI vs KRR equivalence test (tau=0)
    # --------------------------------------------------
    print("\n--- Equivalence Test: LUSI (tau=0) vs KRR ---")

    alpha_no_invariants = fit_lusi(
        X_train,
        y_train,
        lusi_gamma=1.0,
        tau=0.0,
        eta=0.1,
        rbf_gamma=0.5
    )

    lusi_no_invariants_scores = (
        K_test @ alpha_no_invariants
    )

    max_equivalence_difference = np.max(
        np.abs(lusi_no_invariants_scores - krr_scores)
    )

    print(
        "LUSI (tau=0) vs KRR maximum difference:",
        max_equivalence_difference
    )

    print(
        "Numerically equivalent:",
        np.allclose(
            lusi_no_invariants_scores,
            krr_scores,
            rtol=1e-10,
            atol=1e-10
        )
    )

    # --------------------------------------------------
    # 9. Numerical diagnostics
    # --------------------------------------------------
    print("\n--- Diagnostics ---")

    print("Coefficient shape:", alpha.shape)

    print(
        "All coefficients finite:",
        np.isfinite(alpha).all()
    )

    # --------------------------------------------------
    # 10. LUSI tau sensitivity analysis
    # --------------------------------------------------
    print("\n--- LUSI Tau Sensitivity ---")

    tau_values = [
        0.0,
        0.01,
        0.1,
        1.0,
        10.0,
        100.0
    ]

    for tau_value in tau_values:

        alpha_tau = fit_lusi(
            X_train,
            y_train,
            lusi_gamma=1.0,
            tau=tau_value,
            eta=0.1,
            rbf_gamma=0.5
        )

        scores_tau = K_test @ alpha_tau

        predictions_tau = np.where(
            scores_tau >= 0,
            1,
            -1
        )

        accuracy_tau = accuracy_score(
            y_test,
            predictions_tau
        )

        mean_difference = np.mean(
            np.abs(scores_tau - krr_scores)
        )

        disagreements = np.sum(
            predictions_tau != krr_test_pred
        )

        print(
            f"tau={tau_value:>6}: "
            f"accuracy={accuracy_tau:.4f}, "
            f"mean_score_difference={mean_difference:.6f}, "
            f"disagreements={disagreements}"
        )
