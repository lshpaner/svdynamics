"""Check base-kernel PSD numerically on a small sample.

This is a diagnostic, not a proof of positive semidefiniteness.
"""
import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from svdynamics import CompositeKernel

X, _ = make_moons(n_samples=60, noise=0.1, random_state=42)
X = StandardScaler().fit_transform(X)
def negative_linear(A, B):
    return -(A @ B.T)

kernel = CompositeKernel([
    "linear", ("rbf", {"gamma": 0.5}),
    ("poly", {"degree": 2}), negative_linear,
])
for name, eigenvalue in zip(
    ["linear", "rbf", "poly", "negative_linear"],
    kernel.min_eigenvalues(X),
):
    print(f"{name:16} smallest eigenvalue={eigenvalue: .6g}")
print("A negative eigenvalue beyond numerical tolerance indicates an indefinite Gram matrix.")