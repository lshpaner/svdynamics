import numpy as np
import pytest

from svdynamics import AdaptiveCompositeKernel, CompositeKernel, SVDClassifier

rng = np.random.default_rng(0)
X = rng.normal(size=(40, 2))
y = (X[:, 0] * X[:, 1] > 0).astype(int)


def nan_kernel(A, B):
    K = A @ B.T
    K[0, 0] = np.nan
    return K


def inf_kernel(A, B):
    K = A @ B.T
    K[0, 0] = np.inf
    return K


def wrong_shape_kernel(A, B):
    return (A @ B.T)[:, :-1]


def flat_kernel(A, B):
    return (A @ B.T).ravel()


def asymmetric_kernel(A, B):
    # k(x, z) = <x, z> + x_0 is not symmetric in (x, z)
    return A @ B.T + A[:, [0]]


@pytest.mark.parametrize("fn, msg", [
    (nan_kernel, "non-finite"),
    (inf_kernel, "non-finite"),
    (wrong_shape_kernel, "shape"),
    (flat_kernel, "shape"),
])
def test_bad_kernel_outputs_raise(fn, msg):
    with pytest.raises(ValueError, match=msg):
        CompositeKernel(["linear", fn]).fit_scales(X)


@pytest.mark.parametrize("normalize", [True, False])
def test_asymmetric_kernel_rejected_by_fit_scales(normalize):
    with pytest.raises(ValueError, match="not symmetric"):
        CompositeKernel(["linear", asymmetric_kernel], normalize=normalize).fit_scales(X)


def test_asymmetric_kernel_rejected_by_adaptive_fit():
    with pytest.raises(ValueError, match="not symmetric"):
        AdaptiveCompositeKernel(["linear", asymmetric_kernel]).fit(X, y)


@pytest.mark.parametrize("normalize", [True, False])
def test_asymmetric_kernel_rejected_by_classifier(normalize):
    with pytest.raises(ValueError, match="not symmetric"):
        SVDClassifier(
            kernel=CompositeKernel(["linear", asymmetric_kernel], normalize=normalize)
        ).fit(X, y)


@pytest.mark.parametrize("normalize", [True, False])
def test_bad_outputs_rejected_without_normalization(normalize):
    with pytest.raises(ValueError, match="non-finite"):
        CompositeKernel(["linear", nan_kernel], normalize=normalize).fit_scales(X)


def test_builtin_kernels_pass_validation():
    k = CompositeKernel(["linear", "rbf", ("poly", {"degree": 2}), "sigmoid"])
    k.fit_scales(X)
    assert k(X[:5], X).shape == (5, 40)


def test_min_eigenvalues_psd_kernels_nonnegative():
    k = CompositeKernel(["linear", ("rbf", {"gamma": 1.0}), ("poly", {"degree": 2})])
    ev = k.min_eigenvalues(X)
    assert ev.shape == (3,)
    assert np.all(ev > -1e-8)


def test_min_eigenvalues_flags_non_psd():
    # A symmetric but indefinite "kernel": negative of a PSD Gram matrix
    k = CompositeKernel(["linear", lambda A, B: -(A @ B.T)])
    ev = k.min_eigenvalues(X)
    assert ev[1] < -1e-6
