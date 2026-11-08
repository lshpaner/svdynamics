from .logo import *
from .kernels import (
    AdaptiveCompositeKernel,
    CompositeKernel,
)
from .svm import (
    SVDClassifier,
    SVDRegressor,
)


__all__ = [
    "AdaptiveCompositeKernel",
    "CompositeKernel",
    "SVDClassifier",
    "SVDRegressor",
]


# Detailed Documentation

detailed_doc = """
Welcome to svdynamics! Support Vector Dynamics is a lightweight
Python library for building and using mixed (composite) kernels for
support vector machines, designed to integrate with scikit-learn.

It provides a simple and extensible interface for combining multiple
kernel functions into a single weighted kernel. Its estimators work
inside scikit-learn pipelines, cross-validation, grid search, and
probability calibration (each covered by the test suite); formal
estimator-compliance checks (check_estimator) are not yet complete.

svdynamics supports both fixed-weight composite kernels and adaptive
composite kernels whose weights are learned from training data using
kernel-target alignment.

svdynamics focuses on making kernel composition a first-class modeling
primitive for support vector machines without requiring changes to the
underlying scikit-learn API.

PyPI: https://pypi.org/project/svdynamics/
Documentation: https://lshpaner.github.io/svdynamics/

Version: 0.0.1a2
"""


# Assign only the detailed documentation to __doc__
__doc__ = detailed_doc


__version__ = "0.0.1a2"
__author__ = "Leonid Shpaner"
__email__ = "lshpaner@ucla.edu"


def help() -> None:
    """
    Print the svdynamics logo and package overview.

    Call as ``svdynamics.help()``. Python's built-in ``help()`` is left
    unchanged, so ``help(svdynamics)`` shows the standard docstring view.
    """

    print(svdynamics_logo)
    print(detailed_doc)
