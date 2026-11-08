"""Compare decision boundaries of RBF, fixed, and adaptive SVMs."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles, make_moons
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from svdynamics import AdaptiveCompositeKernel, CompositeKernel, SVDClassifier

datasets = {
    "Moons": make_moons(n_samples=250, noise=0.2, random_state=42),
    "Circles": make_circles(n_samples=250, noise=0.1, factor=0.45, random_state=42),
}
specs = ["linear", ("rbf", {"gamma": 1}), ("poly", {"degree": 2, "coef0": 1})]
models = {
    "RBF": SVC(kernel="rbf", gamma=1),
    "Fixed": SVDClassifier(kernel=CompositeKernel(specs, weights=[0.2, 0.6, 0.2])),
    "Adaptive": SVDClassifier(kernel=AdaptiveCompositeKernel(specs)),
}
fig, axes = plt.subplots(len(datasets), len(models), figsize=(12, 7))
for row, (dataset, (X, y)) in enumerate(datasets.items()):
    x0 = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 130)
    x1 = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 130)
    xx, yy = np.meshgrid(x0, x1)
    grid = np.c_[xx.ravel(), yy.ravel()]
    for col, (name, estimator) in enumerate(models.items()):
        model = make_pipeline(StandardScaler(), estimator)
        model.fit(X, y)
        decision = model.decision_function(grid).reshape(xx.shape)
        ax = axes[row, col]
        ax.contourf(xx, yy, decision, levels=20, alpha=0.35)
        ax.contour(xx, yy, decision, levels=[0], colors="black", linewidths=1)
        ax.scatter(X[:, 0], X[:, 1], c=y, s=12, edgecolors="none")
        ax.set_title(f"{dataset}: {name}")
fig.tight_layout()

# Save rather than open a window, so `make` runs unattended.
out = Path("data_output") / "decision_boundaries.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved figure to {out}")