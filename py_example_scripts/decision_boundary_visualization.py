"""Compare decision boundaries of RBF, fixed, and adaptive SVMs."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to file; no window needed
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

# Output folder and resolution are configurable so the same script feeds both
# data_output/ (default, 300 DPI) and the docs (make docs_figures).
parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", default="data_output")
parser.add_argument("--dpi", type=int, default=300)
args = parser.parse_args()

out = Path(args.out_dir) / "decision_boundaries.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
plt.close(fig)
print(f"Saved figure to {out}")