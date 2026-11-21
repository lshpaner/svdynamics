"""Integration tests for the domain-adaptive composite kernel."""
import numpy as np
import pytest
from svdynamics.kernels import DomainAdaptiveCompositeKernel


@pytest.fixture
def training_data():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 4)
    y = (X[:, 0] > 0).astype(int)
    domains = np.repeat([0, 1], 50)
    return X, y, domains


def test_fit_learns_nonuniform_normalized_weights(training_data):
    X, y, domains = training_data
    kernel = DomainAdaptiveCompositeKernel(
        kernels=["linear", "rbf", "poly"], penalty=0.5
    ).fit(X, y, domains)
    assert kernel.weights_.shape == (3,)
    assert np.all(np.isfinite(kernel.weights_))
    assert np.all(kernel.weights_ >= 0)
    assert kernel.weights_.sum() == pytest.approx(1.0)
    assert not np.allclose(kernel.weights_, np.full(3, 1.0 / 3.0))
    assert kernel.alignments_.shape == (3,)
    assert kernel.discrepancies_.shape == (3,)
    assert np.all(kernel.discrepancies_ >= 0)


def test_training_kernel_is_symmetric_and_finite(training_data):
    X, y, domains = training_data
    kernel = DomainAdaptiveCompositeKernel(
        kernels=["linear", "rbf", "poly"], penalty=0.5
    ).fit(X, y, domains)
    K = kernel(X, X)
    assert K.shape == (len(X), len(X))
    assert np.all(np.isfinite(K))
    np.testing.assert_allclose(K, K.T, rtol=1e-10, atol=1e-10)


def test_cross_kernel_is_independent_of_other_query_samples(training_data):
    X, y, domains = training_data
    kernel = DomainAdaptiveCompositeKernel(
        kernels=["linear", "rbf", "poly"], penalty=0.5
    ).fit(X, y, domains)
    X_query = X[:5] + 0.25
    combined = kernel(X_query, X)
    for i in range(len(X_query)):
        np.testing.assert_allclose(
            combined[i:i + 1], kernel(X_query[i:i + 1], X),
            rtol=1e-10, atol=1e-10,
        )


def test_zero_penalty_recovers_alignment_weights(training_data):
    X, y, domains = training_data
    kernel = DomainAdaptiveCompositeKernel(
        kernels=["linear", "rbf", "poly"], penalty=0.0
    ).fit(X, y, domains)
    positive = np.maximum(kernel.alignments_, 0.0)
    np.testing.assert_allclose(kernel.weights_, positive / positive.sum())


def test_invalid_domain_count_rejected(training_data):
    X, y, domains = training_data
    kernel = DomainAdaptiveCompositeKernel(kernels=["linear", "rbf"])
    with pytest.raises(ValueError, match="exactly two"):
        kernel.fit(X, y, np.zeros_like(domains))


def test_mismatched_domain_lengths_rejected(training_data):
    X, y, domains = training_data
    kernel = DomainAdaptiveCompositeKernel(kernels=["linear", "rbf"])
    with pytest.raises(ValueError, match="same number of samples"):
        kernel.fit(X, y, domains[:-1])
