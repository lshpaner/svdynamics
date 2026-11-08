"""Repeated-CV accuracy of single and composite kernels, as box plots.

Figure version of adaptive_kernel_comparison.py. Scaling and adaptive weight
fitting happen inside each fold.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to file; no window needed
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from svdynamics import AdaptiveCompositeKernel, CompositeKernel, SVDClassifier

datasets = {
    "Moons": make_moons(n_samples=300, noise=0.22, random_state=42),
    "Circles": make_circles(n_samples=300, noise=0.12, factor=0.45, random_state=42),
    "Classification": make_classification(
        n_samples=300, n_features=8, n_informative=5, n_redundant=1,
        random_state=42),
}
specs = [("rbf", {"gamma": 0.5}), "linear", ("poly", {"degree": 2, "coef0": 1})]
models = {
    "RBF": SVC(kernel="rbf", gamma=0.5),
    "Linear": SVC(kernel="linear"),
    "Polynomial": SVC(kernel="poly", degree=2, coef0=1),
    "Fixed": SVDClassifier(kernel=CompositeKernel(specs, weights=[0.5, 0.25, 0.25])),
    "Adaptive": SVDClassifier(kernel=AdaptiveCompositeKernel(specs)),
}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

results = {}
for dataset, (X, y) in datasets.items():
    results[dataset] = {}
    for name, estimator in models.items():
        scores = cross_val_score(make_pipeline(StandardScaler(), estimator),
                                 X, y, cv=cv, scoring="accuracy")
        results[dataset][name] = scores
        print(f"{dataset:15} {name:12} {scores.mean():.3f} +/- {scores.std():.3f}")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
for ax, (dataset, series) in zip(axes, results.items()):
    names = list(series)
    ax.boxplot([series[name] for name in names], showmeans=True)
    # Set labels separately: boxplot's label argument was renamed in matplotlib 3.9
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=35)
    ax.set_title(dataset)
    ax.grid(axis="y", alpha=0.25)
axes[0].set_ylabel("Accuracy (repeated stratified CV)")
fig.suptitle("Kernel comparison: 5 folds x 2 repeats, random_state=42")
fig.tight_layout()

# Output folder and resolution are configurable so the same script feeds both
# data_output/ (default, 300 DPI) and the docs (make docs_figures).
parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", default="data_output")
parser.add_argument("--dpi", type=int, default=300)
args = parser.parse_args()

out = Path(args.out_dir) / "repeated_cv_comparison.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
plt.close(fig)
print(f"Saved figure to {out}")
