# Support Vector Dynamics (support-vector-dynamics)

Support Vector Dynamics is a small library that adds first-class mixed (composite) kernels to scikit-learn SVMs.

## Install (editable)
```bash
pip install -e .
```

## Quick Start

```python
import numpy as np
from sklearn.datasets import make_classification
from support_vector_dynamics import CompositeKernel, SVDClassifier

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

## Notes

- This library delegates training to scikit-learn's SVC/SVR.
- Kernel weights are fixed in v0.0.0a0.
- No GPU support yet.