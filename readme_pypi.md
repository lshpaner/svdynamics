<img src="https://raw.githubusercontent.com/lshpaner/svdynamics/refs/heads/main/assets/svd_logo.svg"
     width="280"
     alt="Support Vector Dynamics logo">

<br>

[![PyPI](https://img.shields.io/pypi/v/svdynamics.svg)](https://pypi.org/project/svdynamics/)
[![Downloads](https://pepy.tech/badge/svdynamics)](https://pepy.tech/project/svdynamics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/lshpaner/svdynamics/blob/main/LICENSE.md)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.23872830.svg)](https://doi.org/10.5281/zenodo.23872830)

Welcome to `svdynamics`! Support Vector Dynamics is a lightweight Python
library for building and using mixed (composite) kernels for support vector
machines, designed to integrate with scikit-learn. It provides a simple and
extensible interface for combining multiple kernel functions into a single
weighted kernel. Its estimators work inside scikit-learn pipelines,
cross-validation, grid search, and probability calibration (each covered by
the test suite); formal estimator-compliance checks (`check_estimator`) are not
yet complete.

`svdynamics` focuses on making kernel composition a first-class modeling primitive
for both classification and regression, without requiring any changes to the
underlying scikit-learn API.


## Prerequisites

Before you install `svdynamics`, ensure your system meets the following requirements:

- `Python`: Version `3.8` or higher.

Additionally, `svdynamics` depends on the following packages, which will be
automatically installed:

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


X, y = make_classification(
    n_samples=300,
    n_features=10,
    random_state=0,
)

kernel = CompositeKernel(
    kernels=[
        ("rbf", {"gamma": 0.2}),
        ("linear", {}),
        ("poly", {"degree": 2, "coef0": 1.0}),
    ],
    weights=[0.6, 0.3, 0.1],
    normalize=True,
)

clf = SVDClassifier(
    C=1.0,
    kernel=kernel,
    probability=True,
    random_state=0,
)

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


## 📚 Citing `svdynamics`

If you use `svdynamics` in your research or projects, please consider citing it.

```bibtex
@software{shpaner_2026_23872830,
  author       = {Shpaner, Leonid},
  title        = {{svdynamics: Support Vector Dynamics}},
  year         = 2026,
  publisher    = {Zenodo},
  version      = {0.0.1a2},
  doi          = {10.5281/zenodo.23872830},
  url          = {https://doi.org/10.5281/zenodo.23872830}
}
```


## ⚖️ License

`svdynamics` is distributed under the MIT License. See
[LICENSE](https://github.com/lshpaner/svdynamics/blob/main/LICENSE.MD)
for more information.

--------