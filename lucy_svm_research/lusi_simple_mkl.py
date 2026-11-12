
"""Simple two-kernel MKL using the optimized SVM dual objective.

Research prototype:
- Binary classification on make_moons
- Two RBF component kernels
- Fixed C and gamma
- Continuous simplex-constrained kernel weights
- Bounded scalar optimization of the SVM dual optimum

This implements a basic wrapper-style MKL objective, not the
full published SimpleMKL algorithm.
"""

import numpy as np

from scipy.optimize import minimize_scalar

from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SEED = 42
C = 10.0
GAMMA = 0.5


def component_kernels(x):
    """Compute separate RBF Gram matrices for each feature."""
    return [
        rbf_kernel(
            x[:, [column]],
            x[:, [column]],
            gamma=GAMMA,
        )
        for column in range(x.shape[1])
    ]


def dual_objective(kernel, y):
    """Return the maximized SVM dual objective for a Gram matrix."""
    model = SVC(
        C=C,
        kernel="precomputed",
    )
    model.fit(kernel, y)

    signed_alpha = np.zeros(len(y), dtype=float)

    signed_alpha[model.support_] = model.dual_coef_[0]

    # Signed dual coefficients already contain class labels.
    alpha = np.abs(signed_alpha)

    objective = (
        np.sum(alpha)
        - 0.5 * signed_alpha @ kernel @ signed_alpha
    )

    return float(objective), model


def main():
    x, y = make_moons(
        n_samples=600,
        noise=0.25,
        random_state=SEED,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=SEED,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    train_kernels = component_kernels(x_train)

    def kernel_at(weight):
        return (
            weight * train_kernels[0]
            + (1.0 - weight) * train_kernels[1]
        )

    evaluations = []

    def objective(weight):
        value, _ = dual_objective(
            kernel_at(weight),
            y_train,
        )

        evaluations.append((float(weight), value))
        return value

    initial_value = objective(0.5)

    result = minimize_scalar(
        objective,
        bounds=(0.0, 1.0),
        method="bounded",
        options={
            "xatol": 1e-4,
            "maxiter": 100,
        },
    )

    # Include endpoints because the optimizer may approach a boundary.
    candidates = [
        (0.0, objective(0.0)),
        (float(result.x), objective(result.x)),
        (1.0, objective(1.0)),
    ]

    best_weight, best_objective = min(
        candidates,
        key=lambda item: item[1],
    )

    learned_train_kernel = kernel_at(best_weight)

    _, model = dual_objective(
        learned_train_kernel,
        y_train,
    )

    test_kernels = [
        rbf_kernel(
            x_test[:, [column]],
            x_train[:, [column]],
            gamma=GAMMA,
        )
        for column in range(x_train.shape[1])
    ]

    learned_test_kernel = (
        best_weight * test_kernels[0]
        + (1.0 - best_weight) * test_kernels[1]
    )

    predictions = model.predict(learned_test_kernel)
    scores = model.decision_function(learned_test_kernel)

    print("\n--- SIMPLE MKL: MOONS ---")
    print("Initial weights: [0.5, 0.5]")
    print(
        "Learned weights: [{:.6f}, {:.6f}]".format(
            best_weight,
            1.0 - best_weight,
        )
    )
    print("Weight sum: {:.6f}".format(
        best_weight + (1.0 - best_weight)
    ))
    print("Initial dual objective: {:.6f}".format(
        initial_value
    ))
    print("Final dual objective: {:.6f}".format(
        best_objective
    ))
    print("Optimizer success: {}".format(result.success))
    print("Dual evaluations: {}".format(len(evaluations)))

    print(
        "Test accuracy: {:.6f}".format(
            accuracy_score(y_test, predictions)
        )
    )
    print(
        "Test ROC-AUC: {:.6f}".format(
            roc_auc_score(y_test, scores)
        )
    )

    if best_objective > initial_value + 1e-5:
        raise AssertionError(
            "MKL objective did not improve"
        )

    if not np.isclose(
        best_weight + (1.0 - best_weight),
        1.0,
    ):
        raise AssertionError(
            "Kernel weights do not sum to one"
        )


if __name__ == "__main__":
    main()
