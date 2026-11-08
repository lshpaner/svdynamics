"""Add a user-defined positive-semidefinite kernel to a composite."""
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from svdynamics import CompositeKernel, SVDClassifier

def quadratic_kernel(A, B):
    """PSD polynomial kernel: (1 + x dot z)^2."""
    return (1 + A @ B.T) ** 2

X, y = make_moons(n_samples=250, noise=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42)
kernel = CompositeKernel(
    [quadratic_kernel, ("rbf", {"gamma": 0.5})],
    weights=[0.4, 0.6], normalize=True)
model = make_pipeline(StandardScaler(), SVDClassifier(kernel=kernel))
model.fit(X_train, y_train)
print("Held-out accuracy:", round(model.score(X_test, y_test), 3))