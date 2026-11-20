"""Tests for kernel-space domain discrepancy."""

import numpy as np
import pytest

from svdynamics.domain import kernel_domain_discrepancy


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
