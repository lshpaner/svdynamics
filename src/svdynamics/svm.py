from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.svm import SVC, SVR

from .kernels import (
    AdaptiveCompositeKernel,
    CompositeKernel,
)


KernelArg = Union[
    str,
    CompositeKernel,
    Any,
]


@dataclass
class SVDClassifier(BaseEstimator, ClassifierMixin):
    """
    Thin wrapper around sklearn.svm.SVC with support for composite
    and adaptive composite kernels.

    If kernel is an AdaptiveCompositeKernel, its weights are learned
    from the training data before fitting the SVC.

    If kernel is a CompositeKernel, the estimator uses its callable
    kernel representation.

    Otherwise, the kernel is passed directly to sklearn.svm.SVC.
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

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "SVDClassifier":
        """
        Fit the support vector classifier.

        Adaptive composite kernels learn their weights from the
        training data before the underlying SVC is fitted.
        """

        if isinstance(
            self.kernel,
            AdaptiveCompositeKernel,
        ):
            self.kernel.fit(
                X,
                y,
            )

        if isinstance(
            self.kernel,
            CompositeKernel,
        ):
            k = self.kernel.as_callable()
        else:
            k = self.kernel

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

        self._model.fit(
            X,
            y,
        )

        # Expose sklearn-required fitted attributes.
        self.classes_ = self._model.classes_

        # Useful for sklearn tooling.
        if hasattr(
            self._model,
            "n_features_in_",
        ):
            self.n_features_in_ = self._model.n_features_in_

        return self

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Predict class labels."""

        return self._model.predict(X)

    def decision_function(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the decision function."""

        return self._model.decision_function(X)

    def predict_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Predict class probabilities when probability=True."""

        if not getattr(
            self._model,
            "probability",
            False,
        ):
            raise AttributeError(
                "predict_proba is only available when "
                "probability=True"
            )

        return self._model.predict_proba(X)

    @property
    def support_vectors_(self) -> np.ndarray:
        """Return support vectors."""

        return self._model.support_vectors_

    @property
    def n_support_(self) -> np.ndarray:
        """Return the number of support vectors for each class."""

        return self._model.n_support_


@dataclass
class SVDRegressor(BaseEstimator, RegressorMixin):
    """
    Thin wrapper around sklearn.svm.SVR with support for
    CompositeKernel.
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

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "SVDRegressor":
        """Fit the support vector regressor."""

        if isinstance(
            self.kernel,
            CompositeKernel,
        ):
            k = self.kernel.as_callable()
        else:
            k = self.kernel

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

        self._model.fit(
            X,
            y,
        )

        return self

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Predict continuous target values."""

        return self._model.predict(X)

    @property
    def support_(self) -> np.ndarray:
        """Return indices of support vectors."""

        return self._model.support_