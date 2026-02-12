import builtins
import sys
from .logo import *
from .kernels import CompositeKernel
from .svm import SVDClassifier, SVDRegressor

__all__ = [
    "CompositeKernel",
    "SVDClassifier",
    "SVDRegressor",
]


# Detailed Documentation

detailed_doc = """
Welcome to svdynamics! Support Vector Dynamics is a lightweight, scikit-learn
compatible Python library for building and using mixed (composite) kernels for
support vector machines. It provides a simple and extensible interface for
combining multiple kernel functions into a single weighted kernel, while
remaining fully compatible with existing sklearn pipelines, cross-validation,
and calibration workflows.

svdynamics focuses on making kernel composition a first-class modeling primitive
for both classification and regression, without requiring any changes to the
underlying scikit-learn API.

PyPI: https://pypi.org/project/svdynamics/
Documentation: https://lshpaner.github.io/svdynamics/

Version: 0.0.0a0
"""

# Assign only the detailed documentation to __doc__
__doc__ = detailed_doc


__version__ = "0.0.0a0"
__author__ = "Leonid Shpaner"
__email__ = "lshpaner@ucla.edu"


# Define the custom help function
def custom_help(obj=None):
    """
    Custom help function to dynamically include ASCII art in help() output.
    """
    if (
        obj is None or obj is sys.modules[__name__]
    ):  # When `help()` is called for this module
        print(svdynamics_logo)  # Print ASCII art first
        print(detailed_doc)  # Print the detailed documentation
    else:
        original_help(obj)  # Use the original help for other objects


# Backup the original help function
original_help = builtins.help

# Override the global help function in builtins
builtins.help = custom_help