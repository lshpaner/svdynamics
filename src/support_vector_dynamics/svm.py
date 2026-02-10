from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.svm import SVC, SVR

from .kernels import CompositeKernel


KernelArg = Union[str, CompositeKernel, Any]  # string for sklearn kernels, CompositeKernel, or callable


@dataclass
class SVDClassifier(BaseEstimator, ClassifierMixin):
    """
    SVDClassifier is a thin wrapper around sklearn.svm.SVC that supports CompositeKernel.

    If kernel is a CompositeKernel, the estimator uses a callable kernel.
    Otherwise it passes through to sklearn SVC.
    """

    C: float = 1.0
    kernel: KernelArg = "rbf"
    degree: int = 3
    gamma: Union[str, float] = "scale"
    coef0: float = 0.0
    shrinking: bool = True
    probability: bool = False
    tol: float = 1e-3
    cache_size: float = 200.0
    class_weight: Optional[Dict[Any, float]] = None
    verbose: bool = False
    max_iter: int = -1
    decision_function_shape: str = "ovr"
    break_ties: bool = False
    random_state: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVDClassifier":
        k = self.kernel.as_callable() if isinstance(self.kernel, CompositeKernel) else self.kernel

        self._model = SVC(
            C=self.C,
            kernel=k,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            shrinking=self.shrinking,
            probability=self.probability,
            tol=self.tol,
            cache_size=self.cache_size,
            class_weight=self.class_weight,
            verbose=self.verbose,
            max_iter=self.max_iter,
            decision_function_shape=self.decision_function_shape,
            break_ties=self.break_ties,
            random_state=self.random_state,
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self._model.decision_function(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not getattr(self._model, "probability", False):
            raise AttributeError("predict_proba is only available when probability=True")
        return self._model.predict_proba(X)

    @property
    def support_vectors_(self) -> np.ndarray:
        return self._model.support_vectors_

    @property
    def n_support_(self) -> np.ndarray:
        return self._model.n_support_


@dataclass
class SVDRegressor(BaseEstimator, RegressorMixin):
    """
    SVDRegressor is a thin wrapper around sklearn.svm.SVR that supports CompositeKernel.
    """

    C: float = 1.0
    kernel: KernelArg = "rbf"
    degree: int = 3
    gamma: Union[str, float] = "scale"
    coef0: float = 0.0
    tol: float = 1e-3
    epsilon: float = 0.1
    shrinking: bool = True
    cache_size: float = 200.0
    verbose: bool = False
    max_iter: int = -1

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVDRegressor":
        k = self.kernel.as_callable() if isinstance(self.kernel, CompositeKernel) else self.kernel

        self._model = SVR(
            C=self.C,
            kernel=k,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            tol=self.tol,
            epsilon=self.epsilon,
            shrinking=self.shrinking,
            cache_size=self.cache_size,
            verbose=self.verbose,
            max_iter=self.max_iter,
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def support_(self) -> np.ndarray:
        return self._model.support_
