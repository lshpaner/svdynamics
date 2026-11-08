from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
from sklearn.metrics.pairwise import (
    linear_kernel,
    polynomial_kernel,
    rbf_kernel,
    sigmoid_kernel,
)

from .utils import KernelSpec, ParsedKernel, _validate_weights


SUPPORTED_KERNELS = {"linear", "rbf", "poly", "sigmoid"}


def _parse_kernel_spec(spec: KernelSpec) -> ParsedKernel:
    """Parse a kernel specification into a ParsedKernel."""

    # Callable kernel
    if callable(spec):
        return ParsedKernel(
            kind="callable",
            params={},
            fn=spec,
        )

    # String kernel
    if isinstance(spec, str):
        k = spec.lower().strip()

        if k not in SUPPORTED_KERNELS:
            raise ValueError(
                f"Unsupported kernel '{spec}'. "
                f"Supported: {sorted(SUPPORTED_KERNELS)}"
            )

        return ParsedKernel(
            kind=k,
            params={},
            fn=None,
        )

    # Tuple kernel, for example ("rbf", {"gamma": 0.1})
    if (
        isinstance(spec, tuple)
        and len(spec) == 2
        and isinstance(spec[0], str)
        and isinstance(spec[1], dict)
    ):
        k = spec[0].lower().strip()

        if k not in SUPPORTED_KERNELS:
            raise ValueError(
                f"Unsupported kernel '{spec[0]}'. "
                f"Supported: {sorted(SUPPORTED_KERNELS)}"
            )

        return ParsedKernel(
            kind=k,
            params=dict(spec[1]),
            fn=None,
        )

    raise TypeError(
        "Kernel spec must be one of: "
        "a string ('rbf', 'linear', 'poly', 'sigmoid'), "
        "a tuple like ('rbf', {'gamma': 0.1}), "
        "or a callable kernel(X, Y)->ndarray."
    )


def _compute_kernel(
    parsed: ParsedKernel,
    X: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    """Compute a kernel matrix and check its shape and values."""

    K = _compute_kernel_raw(
        parsed,
        X,
        Y,
    )

    if K.ndim != 2 or K.shape != (X.shape[0], Y.shape[0]):
        raise ValueError(
            f"Kernel '{parsed.kind}' returned shape {K.shape}; "
            f"expected ({X.shape[0]}, {Y.shape[0]})"
        )

    if not np.all(np.isfinite(K)):
        raise ValueError(
            f"Kernel '{parsed.kind}' returned non-finite values "
            "(NaN or inf)"
        )

    return K


def _check_gram_symmetric(
    K: np.ndarray,
    kind: str,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> None:
    """Raise if a training Gram matrix K(X, X) is not symmetric."""

    if not np.allclose(K, K.T, rtol=rtol, atol=atol):
        raise ValueError(
            f"Kernel '{kind}' is not symmetric: K(X, X) != K(X, X)^T. "
            "A valid kernel must satisfy k(x, z) = k(z, x)."
        )


def _compute_kernel_raw(
    parsed: ParsedKernel,
    X: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    """Compute a kernel matrix from a parsed kernel specification."""

    if parsed.kind == "callable":
        assert parsed.fn is not None
        K = parsed.fn(X, Y)
        return np.asarray(K)

    if parsed.kind == "linear":
        return linear_kernel(X, Y)

    if parsed.kind == "rbf":
        return rbf_kernel(
            X,
            Y,
            **parsed.params,
        )

    if parsed.kind == "poly":
        return polynomial_kernel(
            X,
            Y,
            **parsed.params,
        )

    if parsed.kind == "sigmoid":
        return sigmoid_kernel(
            X,
            Y,
            **parsed.params,
        )

    raise RuntimeError(
        f"Unexpected kernel kind: {parsed.kind}"
    )


def _kernel_scale_from_diag(
    Kxx: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Compute a stable scale factor from a kernel diagonal."""

    d = np.diag(Kxx)

    s = float(
        np.sqrt(
            max(
                float(np.mean(d * d)),
                eps,
            )
        )
    )

    if not np.isfinite(s) or s <= 0.0:
        return 1.0

    return s


def _binary_targets(y: np.ndarray) -> np.ndarray:
    """Map any two-class label vector to -1/+1 for alignment."""

    y = np.asarray(y).ravel()
    classes = np.unique(y)

    if classes.shape[0] != 2:
        raise ValueError(
            "AdaptiveCompositeKernel supports binary targets only; "
            f"got {classes.shape[0]} classes"
        )

    # Which class becomes +1 does not matter: y y^T is unchanged
    # when every label flips sign.
    return np.where(y == classes[1], 1.0, -1.0)


def _kernel_target_alignment(
    K: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Compute normalized alignment between a kernel matrix and targets."""

    K = np.asarray(
        K,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    ).ravel()

    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(
            "K must be a square kernel matrix"
        )

    if K.shape[0] != y.shape[0]:
        raise ValueError(
            "K and y must contain the same number of samples"
        )

    target_kernel = np.outer(
        y,
        y,
    )

    numerator = np.sum(
        K * target_kernel
    )

    denominator = (
        np.linalg.norm(
            K,
            ord="fro",
        )
        * np.linalg.norm(
            target_kernel,
            ord="fro",
        )
    )

    if denominator <= eps:
        return 0.0

    return float(
        numerator / denominator
    )


def _alignment_weights(
    kernels: Sequence[np.ndarray],
    y: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute normalized kernel weights from target alignment."""

    if len(kernels) == 0:
        raise ValueError(
            "kernels must be a non-empty sequence"
        )

    alignments = np.asarray(
        [
            _kernel_target_alignment(
                K,
                y,
                eps=eps,
            )
            for K in kernels
        ],
        dtype=float,
    )

    # Negative alignment does not contribute to the mixture.
    alignments = np.maximum(
        alignments,
        0.0,
    )

    total = alignments.sum()

    # Fall back to uniform weights if no kernel has
    # meaningful positive alignment.
    if total <= eps:
        return (
            np.ones(
                len(kernels),
                dtype=float,
            )
            / len(kernels)
        )

    return alignments / total


@dataclass
class CompositeKernel:
    """
    Weighted combination of multiple base kernels.

    K(X, Y) = sum_m w_m * K_m(X, Y)

    Parameters
    ----------
    kernels:
        Sequence of kernel specifications. Each element can be:

        - "rbf", "linear", "poly", or "sigmoid"
        - ("rbf", {"gamma": 0.1}), etc.
        - callable kernel(X, Y) -> ndarray

    weights:
        Optional weights for the kernels. If None, uniform weights
        are used. Weights are normalized to sum to 1.

    normalize:
        If True, each base kernel is rescaled using a scalar derived
        from K(X, X).

    eps:
        Small constant used for numerical stability.
    """

    kernels: Sequence[KernelSpec]
    weights: Optional[Sequence[float]] = None
    normalize: bool = True
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if self.kernels is None or len(self.kernels) == 0:
            raise ValueError(
                "kernels must be a non-empty sequence"
            )

        self._parsed: List[ParsedKernel] = [
            _parse_kernel_spec(spec)
            for spec in self.kernels
        ]

        self._weights: np.ndarray = _validate_weights(
            len(self._parsed),
            self.weights,
        )

    @property
    def parsed_kernels(self) -> List[ParsedKernel]:
        """Return parsed kernel specifications."""

        return list(self._parsed)

    @property
    def normalized_weights(self) -> np.ndarray:
        """Return normalized kernel weights."""

        return self._weights.copy()

    def min_eigenvalues(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Smallest eigenvalue of each base kernel's Gram matrix on X.

        Optional diagnostic for positive semidefiniteness (PSD). A valid
        kernel gives values >= 0 up to rounding (about -1e-8 relative to
        the largest eigenvalue). Sigmoid kernels are not PSD in general and
        may return clearly negative values. Cost is O(n^3) per kernel, so
        use a subsample for large X.
        """

        X = np.asarray(X)

        return np.asarray(
            [
                float(
                    np.linalg.eigvalsh(
                        _compute_kernel(
                            pk,
                            X,
                            X,
                        )
                    )[0]
                )
                for pk in self._parsed
            ],
            dtype=float,
        )

    def fit_scales(
        self,
        X: np.ndarray,
    ) -> "CompositeKernel":
        """
        Compute and store per-kernel normalization scales from
        training data.

        Once stored, every later call uses these scales, so a point's
        kernel values (and the model's predictions) do not depend on
        which other points are passed alongside it.
        """

        X = np.asarray(X)

        scales = []

        for pk in self._parsed:
            # Validate every training Gram matrix, whether or not it is
            # used for normalization.
            Kxx = _compute_kernel(
                pk,
                X,
                X,
            )
            _check_gram_symmetric(
                Kxx,
                pk.kind,
            )
            scales.append(
                _kernel_scale_from_diag(
                    Kxx,
                    eps=self.eps,
                )
                if self.normalize
                else 1.0
            )

        self.scales_ = np.asarray(
            scales,
            dtype=float,
        )

        return self

    def as_callable(
        self,
    ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """Return the composite kernel as a callable."""

        def _k(
            X: np.ndarray,
            Y: np.ndarray,
        ) -> np.ndarray:
            return self(X, Y)

        return _k

    def __call__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> np.ndarray:
        """Compute the composite kernel matrix."""

        X = np.asarray(X)
        Y = np.asarray(Y)

        scales: List[float] = []

        if getattr(self, "scales_", None) is not None:
            # Training-derived scales (set by fit_scales).
            scales = list(self.scales_)
        elif self.normalize:
            # Fallback for standalone use before fit_scales():
            # scales come from the first argument.
            if Y is not X:
                warnings.warn(
                    "CompositeKernel called before fit_scales(); "
                    "normalization scales are computed from the first "
                    "argument, so results depend on which points are "
                    "passed together. Call fit_scales(X_train) first.",
                    UserWarning,
                    stacklevel=2,
                )
            for pk in self._parsed:
                Kxx = _compute_kernel(
                    pk,
                    X,
                    X,
                )

                if Kxx.shape[0] != Kxx.shape[1]:
                    raise ValueError(
                        "Expected square K(X, X) during normalization"
                    )

                scales.append(
                    _kernel_scale_from_diag(
                        Kxx,
                        eps=self.eps,
                    )
                )
        else:
            scales = [1.0] * len(self._parsed)

        K_total = None

        for w, s, pk in zip(
            self._weights,
            scales,
            self._parsed,
        ):
            K = _compute_kernel(
                pk,
                X,
                Y,
            )

            if K_total is None:
                K_total = (w / s) * K
            else:
                K_total += (w / s) * K

        assert K_total is not None

        return np.asarray(
            K_total,
            dtype=float,
        )


@dataclass
class AdaptiveCompositeKernel(CompositeKernel):
    """
    Composite kernel with weights learned from training data.

    Kernel weights are estimated using normalized kernel-target
    alignment.

    Parameters
    ----------
    kernels:
        Sequence of kernel specifications.

    weights:
        Initial kernel weights. These are replaced by learned
        weights when fit() is called.

    normalize:
        If True, rescale individual kernels when constructing the
        composite kernel.

    eps:
        Small constant used for numerical stability.
    """

    weights: Optional[Sequence[float]] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "AdaptiveCompositeKernel":
        """Learn kernel weights from training data."""

        X = np.asarray(X)

        y = _binary_targets(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X and y must contain the same number of samples"
            )

        kernel_matrices = [
            _compute_kernel(
                pk,
                X,
                X,
            )
            for pk in self._parsed
        ]

        for pk, K in zip(
            self._parsed,
            kernel_matrices,
        ):
            _check_gram_symmetric(
                K,
                pk.kind,
            )

        self.alignments_ = np.asarray(
            [
                _kernel_target_alignment(
                    K,
                    y,
                    eps=self.eps,
                )
                for K in kernel_matrices
            ],
            dtype=float,
        )

        self._weights = _alignment_weights(
            kernel_matrices,
            y,
            eps=self.eps,
        )

        self.weights_ = self._weights.copy()

        self.fit_scales(X)

        return self