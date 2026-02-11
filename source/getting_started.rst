Getting started
===============

Installation
------------

Install from source (current development version):

.. code-block:: bash

   pip install -e .


Basic usage
-----------

The central concept in svdynamics is the CompositeKernel object.

A composite kernel represents a weighted sum of multiple base kernels.

Example:

.. code-block:: python

   from svdynamics import CompositeKernel, SVDClassifier
   from sklearn.datasets import make_classification

   X, y = make_classification(
       n_samples=300,
       n_features=10,
       random_state=0
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

   y_pred = clf.predict(X)
   y_prob = clf.predict_proba(X)


How it works
------------

Composite kernels are implemented as callable kernels passed directly to
scikit-learn's SVC / SVR.

The resulting kernel matrix is computed as a weighted sum of the individual
kernel matrices:

.. math::

   K(x, x') = \sum_{i=1}^{m} w_i \, K_i(x, x')

This makes svdynamics fully compatible with sklearn internals.
