
"""Kernel construction for the MAGIC LUSI benchmark.

Research reference:
Liu, Fokoue, and Krutz (2025), Section 6.3.

This is an independent experimental implementation,
not an exact reproduction of their unpublished settings.
"""

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler


BASE_FEATURES = ["fLength", "fWidth"]


def split_features(X_train, X_test):
    """Separate baseline and increment features without leakage."""
    increment_features = [
        col for col in X_train.columns
        if col not in BASE_FEATURES
    ]

    base_scaler = StandardScaler()
    inc_scaler = StandardScaler()

    X_base_train = base_scaler.fit_transform(
        X_train[BASE_FEATURES]
    )
    X_base_test = base_scaler.transform(
        X_test[BASE_FEATURES]
    )

    X_inc_train = inc_scaler.fit_transform(
        X_train[increment_features]
    )
    X_inc_test = inc_scaler.transform(
        X_test[increment_features]
    )

    return (
        X_base_train,
        X_base_test,
        X_inc_train,
        X_inc_test,
    )


def construct_kernels(
    X_base_train,
    X_base_test,
    X_inc_train,
    X_inc_test,
    gamma_base=0.5,
    gamma_inc=0.5,
):
    """Construct baseline and increment RBF Gram matrices."""

    K_base_train = rbf_kernel(
        X_base_train,
        gamma=gamma_base,
    )

    K_base_test = rbf_kernel(
        X_base_test,
        X_base_train,
        gamma=gamma_base,
    )

    K_inc_train = rbf_kernel(
        X_inc_train,
        gamma=gamma_inc,
    )

    K_inc_test = rbf_kernel(
        X_inc_test,
        X_inc_train,
        gamma=gamma_inc,
    )

    return {
        "K_base_train": K_base_train,
        "K_base_test": K_base_test,
        "K_inc_train": K_inc_train,
        "K_inc_test": K_inc_test,
    }


def validate_kernels(kernels):
    """Check Gram matrix symmetry and positive semidefiniteness."""

    for name in ("K_base_train", "K_inc_train"):
        K = kernels[name]

        assert np.allclose(K, K.T, atol=1e-10)
        assert np.isfinite(K).all()

        min_eigenvalue = np.linalg.eigvalsh(K).min()

        print(
            "{}: shape={}, min eigenvalue={:.3e}".format(
                name,
                K.shape,
                min_eigenvalue,
            )
        )

        assert min_eigenvalue >= -1e-8
