"""Plot adaptive kernel weights on three different binary datasets."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles, make_moons, make_classification
from sklearn.preprocessing import StandardScaler
from svdynamics import AdaptiveCompositeKernel

datasets = {
    "Moons": make_moons(n_samples=250, noise=0.2, random_state=42),
    "Circles": make_circles(n_samples=250, noise=0.1, factor=0.45, random_state=42),
    "Linear-ish": make_classification(n_samples=250, n_features=2,
        n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42),
}
labels = ["Linear", "RBF", "Polynomial"]
specs = ["linear", ("rbf", {"gamma": 0.5}),
         ("poly", {"degree": 2, "coef0": 1})]
weights = []
for name, (X, y) in datasets.items():
    X = StandardScaler().fit_transform(X)
    kernel = AdaptiveCompositeKernel(specs).fit(X, y)
    weights.append(kernel.weights_)
    print(name, "weights:", np.round(kernel.weights_, 3),
          "alignments:", np.round(kernel.alignments_, 3))
fig, ax = plt.subplots(figsize=(8, 4))
positions = np.arange(len(datasets))
bottom = np.zeros(len(datasets))
for i, label in enumerate(labels):
    values = np.array(weights)[:, i]
    ax.bar(positions, values, bottom=bottom, label=label)
    bottom += values
ax.set_xticks(positions, list(datasets))
ax.set_ylabel("Learned mixture weight")
ax.set_ylim(0, 1)
ax.legend()
fig.tight_layout()

# Save rather than open a window, so `make` runs unattended.
out = Path("data_output") / "kernel_weights.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved figure to {out}")