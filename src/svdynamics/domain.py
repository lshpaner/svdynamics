"""Domain-discrepancy diagnostics for precomputed kernel matrices."""

from __future__ import annotations

import numpy as np


def kernel_domain_discrepancy(K: np.ndarray, domains: np.ndarray) -> float:
    """Compute biased squared MMD between two domains in kernel space.

    Parameters
    ----------
    K:
        Square, symmetric training Gram matrix. A positive-semidefinite
        kernel is assumed.
    domains:
        One domain label per training sample, with exactly two distinct
        labels and at least one sample from each domain.

    Returns
    -------
    float
        Squared maximum mean discrepancy (MMD), estimated as
        mean(K_AA) + mean(K_BB) - 2 * mean(K_AB).
        The biased estimator includes diagonal entries and is
        nonnegative for positive-semidefinite kernels, up to roundoff.

    Notes
    -----
    This is a diagnostic, not a domain-adaptation optimizer. Compute
    it on training data only when selecting kernel weights.
    """
    K = np.asarray(K, dtype=float)
    domains = np.asarray(domains).ravel()

    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square training Gram matrix")
    if K.shape[0] != domains.shape[0]:
        raise ValueError("K and domains must have the same number of samples")
    if K.shape[0] < 2 or not np.all(np.isfinite(K)):
        raise ValueError("K must contain at least two samples and finite values")
    if not np.allclose(K, K.T, rtol=1e-6, atol=1e-8):
        raise ValueError("K must be symmetric")

    labels, inverse = np.unique(domains, return_inverse=True)
    if len(labels) != 2:
        raise ValueError("domains must contain exactly two distinct labels")

    a = inverse == 0
    b = inverse == 1
    mmd2 = (
        K[np.ix_(a, a)].mean()
        + K[np.ix_(b, b)].mean()
        - 2.0 * K[np.ix_(a, b)].mean()
    )
    return float(max(0.0, mmd2))
