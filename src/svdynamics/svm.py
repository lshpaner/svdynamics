from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.svm import SVC, SVR
from sklearn.utils.validation import check_is_fitted

from .kernels import AdaptiveCompositeKernel, CompositeKernel


KernelArg = Union[str, CompositeKernel, Any]


@dataclass
class SVDClassifier(BaseEstimator, ClassifierMixin):
    """SVC wrapper supporting fixed and adaptive composite kernels."""

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

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "_model")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVDClassifier":
        if isinstance(self.kernel, CompositeKernel):
            kernel = copy.deepcopy(self.kernel)
            if isinstance(kernel, AdaptiveCompositeKernel):
                kernel.fit(X, y)
            else:
                kernel.fit_scales(X)
            fitted_kernel = kernel
            k = kernel.as_callable()
        else:
            fitted_kernel = self.kernel
            k = self.kernel

        model = SVC(
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
        model.fit(X, y)
        self.kernel_ = fitted_kernel
        self._model = model
        self.classes_ = model.classes_
        # SVC/SVR gained n_iter_ in scikit-learn 1.1; the package supports 1.0+.
        if hasattr(model, "n_iter_"):
            self.n_iter_ = model.n_iter_
        # SVC/SVR skip n_features_in_ when the kernel is a callable
        # (every composite kernel), so fall back to the input's width.
        if hasattr(model, "n_features_in_"):
            self.n_features_in_ = model.n_features_in_
        else:
            self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        return self._model.predict(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        return self._model.decision_function(X)

    @property
    def predict_proba(self):
        """Available only when probability estimation is enabled."""
        if not self.probability:
            raise AttributeError(
                "predict_proba is only available when probability=True"
            )
        return self._predict_proba

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        return self._model.predict_proba(X)

    @property
    def predict_log_proba(self):
        """Available only when probability estimation is enabled."""
        if not self.probability:
            raise AttributeError(
                "predict_log_proba is only available when probability=True"
            )
        return self._predict_log_proba

    def _predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        return self._model.predict_log_proba(X)

    @property
    def support_(self) -> np.ndarray:
        check_is_fitted(self)
        return self._model.support_

    @property
    def support_vectors_(self) -> np.ndarray:
        check_is_fitted(self)
        return self._model.support_vectors_

    @property
    def n_support_(self) -> np.ndarray:
        check_is_fitted(self)
        return self._model.n_support_


@dataclass
class SVDRegressor(BaseEstimator, RegressorMixin):
    """SVR wrapper supporting fixed composite kernels."""

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

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "_model")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVDRegressor":
        if isinstance(self.kernel, AdaptiveCompositeKernel):
            raise TypeError(
                "AdaptiveCompositeKernel learns weights from binary class "
                "labels and is not supported for regression; use "
                "CompositeKernel with fixed weights."
            )

        if isinstance(self.kernel, CompositeKernel):
            kernel = copy.deepcopy(self.kernel)
            kernel.fit_scales(X)
            fitted_kernel = kernel
            k = kernel.as_callable()
        else:
            fitted_kernel = self.kernel
            k = self.kernel

        model = SVR(
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
        model.fit(X, y)
        self.kernel_ = fitted_kernel
        self._model = model
        # SVC/SVR gained n_iter_ in scikit-learn 1.1; the package supports 1.0+.
        if hasattr(model, "n_iter_"):
            self.n_iter_ = model.n_iter_
        # SVC/SVR skip n_features_in_ when the kernel is a callable
        # (every composite kernel), so fall back to the input's width.
        if hasattr(model, "n_features_in_"):
            self.n_features_in_ = model.n_features_in_
        else:
            self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        return self._model.predict(X)

    @property
    def support_(self) -> np.ndarray:
        check_is_fitted(self)
        return self._model.support_

    @property
    def support_vectors_(self) -> np.ndarray:
        check_is_fitted(self)
        return self._model.support_vectors_

    @property
    def n_support_(self) -> np.ndarray:
        check_is_fitted(self)
        return self._model.n_support_