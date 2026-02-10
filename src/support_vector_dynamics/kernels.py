from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel, sigmoid_kernel

from .utils import ParsedKernel, KernelSpec, _validate_weights


SUPPORTED_KERNELS = {"linear", "rbf", "poly", "sigmoid"}


def _parse_kernel_spec(spec: KernelSpec) -> ParsedKernel:
    # Callable kernel
    if callable(spec):
        return ParsedKernel(kind="callable", params={}, fn=spec)

    # String kernel
    if isinstance(spec, str):
        k = spec.lower().strip()
        if k not in SUPPORTED_KERNELS:
            raise ValueError(f"Unsupported kernel '{spec}'. Supported: {sorted(SUPPORTED_KERNELS)}")
        return ParsedKernel(kind=k, params={}, fn=None)

    # Tuple kernel ("rbf", {...})
    if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], str) and isinstance(spec[1], dict):
        k = spec[0].lower().strip()
        if k not in SUPPORTED_KERNELS:
            raise ValueError(f"Unsupported kernel '{spec[0]}'. Supported: {sorted(SUPPORTED_KERNELS)}")
        return ParsedKernel(kind=k, params=dict(spec[1]), fn=None)

    raise TypeError(
        "Kernel spec must be one of: "
        "a string ('rbf', 'linear', 'poly', 'sigmoid'), "
        "a tuple like ('rbf', {'gamma': 0.1}), "
        "or a callable kernel(X, Y)->ndarray."
    )


def _compute_kernel(parsed: ParsedKernel, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    if parsed.kind == "callable":
        assert parsed.fn is not None
        K = parsed.fn(X, Y)
        return np.asarray(K)

    if parsed.kind == "linear":
        return linear_kernel(X, Y)

    if parsed.kind == "rbf":
        return rbf_kernel(X, Y, **parsed.params)

    if parsed.kind == "poly":
        return polynomial_kernel(X, Y, **parsed.params)

    if parsed.kind == "sigmoid":
        return sigmoid_kernel(X, Y, **parsed.params)

    raise RuntimeError(f"Unexpected kernel kind: {parsed.kind}")


def _kernel_scale_from_diag(Kxx: np.ndarray, eps: float = 1e-12) -> float:
    # Kxx is square kernel matrix
    d = np.diag(Kxx)
    # Use mean of diagonal as a stable scalar, avoid zeros
    s = float(np.sqrt(max(float(np.mean(d * d)), eps)))
    if not np.isfinite(s) or s <= 0.0:
        return 1.0
    return s


@dataclass
class CompositeKernel:
    """
    CompositeKernel builds a callable kernel that is a weighted sum of base kernels.

    K(X, Y) = sum_m w_m * K_m(X, Y)

    Parameters
    ----------
    kernels:
        List of kernel specs. Each element can be:
        - "rbf" | "linear" | "poly" | "sigmoid"
        - ("rbf", {"gamma": 0.1}) etc.
        - callable kernel(X, Y) -> ndarray
    weights:
        Optional weights for kernels. If None, uniform weights are used.
        Weights are normalized to sum to 1.
    normalize:
        If True, each base kernel is rescaled by a scalar derived from K(X, X)
        to reduce dominance due to scale.
    eps:
        Small constant used for numerical stability.
    """

    kernels: Sequence[KernelSpec]
    weights: Optional[Sequence[float]] = None
    normalize: bool = True
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if self.kernels is None or len(self.kernels) == 0:
            raise ValueError("kernels must be a non-empty sequence")

        self._parsed: List[ParsedKernel] = [_parse_kernel_spec(s) for s in self.kernels]
        self._weights: np.ndarray = _validate_weights(len(self._parsed), self.weights)

    @property
    def parsed_kernels(self) -> List[ParsedKernel]:
        return list(self._parsed)

    @property
    def normalized_weights(self) -> np.ndarray:
        return self._weights.copy()

    def as_callable(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        def _k(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
            return self(X, Y)

        return _k

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        Y = np.asarray(Y)

        # Compute normalization scalars using K(X, X) for each kernel if requested
        scales: List[float] = []
        if self.normalize:
            for pk in self._parsed:
                Kxx = _compute_kernel(pk, X, X)
                if Kxx.shape[0] != Kxx.shape[1]:
                    raise ValueError("Expected square K(X, X) during normalization")
                scales.append(_kernel_scale_from_diag(Kxx, eps=self.eps))
        else:
            scales = [1.0] * len(self._parsed)

        # Combine kernels
        K_total = None
        for w, s, pk in zip(self._weights, scales, self._parsed):
            K = _compute_kernel(pk, X, Y)
            if K_total is None:
                K_total = (w / s) * K
            else:
                K_total += (w / s) * K

        assert K_total is not None
        return np.asarray(K_total, dtype=float)
