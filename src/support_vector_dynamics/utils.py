from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


KernelSpec = Union[
    str,  # "rbf", "linear", "poly", "sigmoid"
    Tuple[str, Dict[str, Any]],  # ("rbf", {"gamma": 0.1})
    Any,  # callable kernel(X, Y) -> ndarray
]


@dataclass(frozen=True)
class ParsedKernel:
    kind: str  # "rbf", "linear", "poly", "sigmoid", or "callable"
    params: Dict[str, Any]
    fn: Optional[Any] = None  # for callable kernels


def _as_list(x: Optional[Sequence[Any]]) -> List[Any]:
    if x is None:
        return []
    return list(x)


def _validate_weights(num_kernels: int, weights: Optional[Sequence[float]]) -> np.ndarray:
    if weights is None:
        w = np.ones(num_kernels, dtype=float)
        return w / w.sum()

    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.shape[0] != num_kernels:
        raise ValueError(f"weights must be length {num_kernels}, got shape {w.shape}")

    if np.any(~np.isfinite(w)):
        raise ValueError("weights must be finite")

    if np.allclose(w.sum(), 0.0):
        raise ValueError("weights sum to 0")

    return w / w.sum()


def _validate_square_kernel_matrix(K: np.ndarray, name: str = "K") -> None:
    if not isinstance(K, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if K.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {K.ndim}D")
    if K.shape[0] == 0 or K.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty, got shape {K.shape}")
