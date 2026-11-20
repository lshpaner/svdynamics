"""Tests for kernel-space domain discrepancy."""

import numpy as np
import pytest

from svdynamics.domain import domain_adaptive_weights, kernel_domain_discrepancy


def test_identical_feature_means_have_zero_linear_mmd():
    X = np.array([[0.0], [2.0], [0.0], [2.0]])
    K = X @ X.T
    domains = np.array(["A", "A", "B", "B"])
    assert kernel_domain_discrepancy(K, domains) == pytest.approx(0.0)


def test_separated_domains_linear_kernel():
    X = np.array([[0.0], [0.0], [2.0], [2.0]])
    K = X @ X.T
    assert kernel_domain_discrepancy(K, [0, 0, 1, 1]) == pytest.approx(4.0)


def test_manual_mmd_includes_diagonal():
    K = np.eye(4)
    domains = [0, 0, 1, 1]
    # Within-domain means are 1/2 each; cross-domain mean is zero.
    assert kernel_domain_discrepancy(K, domains) == pytest.approx(1.0)


def test_domain_label_permutation_invariant():
    K = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.2], [0.0, 0.2, 1.0]])
    assert kernel_domain_discrepancy(K, ["site1", "site2", "site1"]) == pytest.approx(
        kernel_domain_discrepancy(K, ["site2", "site1", "site2"])
    )


@pytest.mark.parametrize(
    "K, domains, message",
    [
        (np.ones((2, 3)), [0, 1], "square"),
        (np.eye(3), [0, 1], "same number"),
        (np.eye(1), [0], "at least two"),
        (np.array([[1.0, np.nan], [np.nan, 1.0]]), [0, 1], "finite"),
        (np.array([[1.0, 2.0], [0.0, 1.0]]), [0, 1], "symmetric"),
        (np.eye(3), [0, 0, 0], "exactly two"),
        (np.eye(3), [0, 1, 2], "exactly two"),
    ],
)
def test_invalid_inputs(K, domains, message):
    with pytest.raises(ValueError, match=message):
        kernel_domain_discrepancy(K, domains)



def test_domain_weights_no_penalty_follow_positive_alignment():
    weights = domain_adaptive_weights([0.2, 0.6], [1.0, 3.0], penalty=0.0)
    np.testing.assert_allclose(weights, [0.25, 0.75])


def test_domain_weights_penalize_high_discrepancy():
    weights = domain_adaptive_weights([0.6, 0.6], [0.0, 2.0], penalty=0.3)
    np.testing.assert_allclose(weights, [2.0 / 3.0, 1.0 / 3.0])


def test_domain_weights_zero_discrepancy():
    weights = domain_adaptive_weights([0.2, 0.8], [0.0, 0.0])
    np.testing.assert_allclose(weights, [0.2, 0.8])


def test_domain_weights_uniform_fallback():
    weights = domain_adaptive_weights([0.1, 0.1], [1.0, 1.0], penalty=1.0)
    np.testing.assert_allclose(weights, [0.5, 0.5])


@pytest.mark.parametrize(
    "alignments, discrepancies, penalty, message",
    [
        ([], [], 0.5, "nonempty"),
        ([0.2], [0.1, 0.2], 0.5, "matching"),
        ([[0.2]], [[0.1]], 0.5, "1D"),
        ([np.nan], [0.1], 0.5, "finite"),
        ([0.2], [-0.1], 0.5, "nonnegative"),
        ([0.2], [0.1], -1.0, "nonnegative"),
        ([0.2], [0.1], np.inf, "finite"),
    ],
)
def test_domain_weights_invalid_inputs(alignments, discrepancies, penalty, message):
    with pytest.raises(ValueError, match=message):
        domain_adaptive_weights(alignments, discrepancies, penalty=penalty)
