<img src="https://raw.githubusercontent.com/lshpaner/svdynamics/refs/heads/main/assets/svd_logo.svg" height="250" style="border: none; outline: none; box-shadow: none;" oncontextmenu="return false;">

Welcome to `svdynamics`! Support Vector Dynamics is a lightweight, scikit-learn
compatible Python library for building and using mixed (composite) kernels for
support vector machines. It provides a simple and extensible interface for
combining multiple kernel functions into a single weighted kernel, while
remaining fully compatible with existing sklearn pipelines, cross-validation,
and calibration workflows.

`svdynamics` focuses on making kernel composition a first-class modeling primitive
for both classification and regression, without requiring any changes to the
underlying scikit-learn API.

## Prerequisites

Before you install `svdynamics`, ensure your system meets the following requirements:

- `Python`: Version `3.8` or higher.


Additionally, `svdynamics` depends on the following packages, which will be automatically installed:

- `numpy`: version `1.21` or higher
- `scikit-learn`: version `1.0` or higher


## 💾 Installation

To install `svdynamics`, simply run the following command in your terminal:


```bash
pip install svdynamics
```

## Quick Start

```python
import numpy as np
from sklearn.datasets import make_classification
from svdynamics import CompositeKernel, SVDClassifier

X, y = make_classification(n_samples=300, n_features=10, random_state=0)

kernel = CompositeKernel(
    kernels=[
        ("rbf", {"gamma": 0.2}),
        ("linear", {}),
        ("poly", {"degree": 2, "coef0": 1.0}),
    ],
    weights=[0.6, 0.3, 0.1],
    normalize=True,
)

clf = SVDClassifier(C=1.0, kernel=kernel, probability=True, random_state=0)
clf.fit(X, y)
proba = clf.predict_proba(X[:5])
pred = clf.predict(X[:5])

print(proba)
print(pred)
```

## 📄 Official Documentation

https://lshpaner.github.io/svdynamics


## 🌐 Authors' Website

1. [Leon Shpaner](https://www.leonshpaner.com)

## ⚖️ License

`svdynamics` is distributed under the MIT License. See [LICENSE](https://github.com/lshpaner/svdynamics/blob/main/LICENSE.md) for more information.