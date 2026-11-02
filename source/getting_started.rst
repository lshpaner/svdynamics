Getting started
===============

Installation
------------

Install the latest release from PyPI:

.. code-block:: bash

   pip install svdynamics


Basic usage
-----------

The central concept in ``svdynamics`` is the ``CompositeKernel`` object.

A composite kernel represents a weighted sum of multiple base kernels. This
allows different kernel functions to be combined while retaining the standard
scikit-learn SVM interface.

Example:

.. code-block:: python

   from svdynamics import CompositeKernel, SVDClassifier
   from sklearn.datasets import make_classification

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

   y_pred = clf.predict(X)
   y_prob = clf.predict_proba(X)


How composite kernels work
--------------------------

Composite kernels are implemented as callable kernels passed directly to
scikit-learn's SVC / SVR.

The resulting kernel matrix is computed as a weighted sum of the individual
kernel matrices:

.. math::

   K(x, x') = \sum_{i=1}^{m} w_i \, K_i(x, x')

where :math:`K_i` represents the i-th base kernel and :math:`w_i` represents
its corresponding weight.

With ``CompositeKernel``, these weights are specified directly by the user.
For example:

.. code-block:: python

   weights=[0.6, 0.3, 0.1]

This makes ``svdynamics`` compatible with scikit-learn's underlying SVM
estimators while providing a convenient interface for combining kernels.


Adaptive composite kernels
--------------------------

``svdynamics`` also provides ``AdaptiveCompositeKernel`` for learning kernel
weights directly from the training data.

Instead of manually specifying the contribution of each kernel,
``AdaptiveCompositeKernel`` evaluates the relationship between each candidate
kernel matrix and the training target. These relationships are then used to
derive the composite kernel weights.

Example:

.. code-block:: python

   from sklearn.datasets import make_circles
   from sklearn.model_selection import train_test_split
   from svdynamics import AdaptiveCompositeKernel, SVDClassifier

   X, y = make_circles(
       n_samples=500,
       noise=0.08,
       factor=0.4,
       random_state=0,
   )

   X_train, X_test, y_train, y_test = train_test_split(
       X,
       y,
       test_size=0.25,
       random_state=0,
       stratify=y,
   )

   kernel = AdaptiveCompositeKernel(
       kernels=[
           ("linear", {}),
           ("rbf", {"gamma": 1.0}),
           ("poly", {"degree": 2, "coef0": 1.0}),
       ],
       normalize=True,
   )

   clf = SVDClassifier(
       C=1.0,
       kernel=kernel,
   )

   clf.fit(X_train, y_train)

   y_pred = clf.predict(X_test)

During fitting, the kernel weights are learned automatically from the training
data rather than supplied manually.


Kernel-target alignment
-----------------------

Adaptive kernel weighting is based on kernel-target alignment.

For a kernel matrix :math:`K` and target kernel :math:`K_y`, alignment is
computed as:

.. math::

   A(K, K_y) =
   \frac{
       \langle K, K_y \rangle_F
   }{
       \|K\|_F \, \|K_y\|_F
   }

where :math:`\langle \cdot, \cdot \rangle_F` denotes the Frobenius inner
product and :math:`\|\cdot\|_F` denotes the Frobenius norm.

For binary classification, the target kernel is constructed from the class
labels as:

.. math::

   K_y = y y^T

The alignment score measures how strongly the structure induced by a candidate
kernel corresponds to the structure of the training labels.

Higher positive alignment indicates greater agreement between the candidate
kernel and the target structure.


Learned kernel weights
----------------------

After fitting an ``AdaptiveCompositeKernel``, the alignment scores and learned
weights can be inspected directly:

.. code-block:: python

   print(kernel.alignments_)
   print(kernel.weights_)

``alignments_`` contains the kernel-target alignment score associated with each
candidate kernel.

``weights_`` contains the corresponding non-negative normalized kernel weights
used by the fitted composite kernel.

For example, a fitted model may produce:

.. code-block:: text

   alignments_: [0.0004, 0.1481, 0.0195]
   weights_:    [0.0021, 0.8802, 0.1177]

In this example, the RBF kernel receives the largest weight because it exhibits
the strongest alignment with the target structure.

The resulting adaptive composite kernel can be written as:

.. math::

   K_{\mathrm{adaptive}}(x, x')
   =
   \sum_{i=1}^{m} \hat{w}_i K_i(x, x')

where :math:`\hat{w}_i` denotes a weight learned from the training data.


Fixed vs. adaptive weighting
----------------------------

``CompositeKernel`` and ``AdaptiveCompositeKernel`` provide two approaches to
kernel composition.

``CompositeKernel``
   Uses kernel weights specified directly by the user.

``AdaptiveCompositeKernel``
   Learns kernel weights from the training data using kernel-target alignment.

The original fixed-weight interface therefore remains available when kernel
contributions are known in advance, while adaptive weighting provides a
data-driven alternative.


Using the fitted classifier
---------------------------

Once fitted, an ``SVDClassifier`` using an adaptive kernel behaves like the
standard classifier interface:

.. code-block:: python

   y_pred = clf.predict(X_test)

If probability estimation is enabled:

.. code-block:: python

   clf = SVDClassifier(
       C=1.0,
       kernel=kernel,
       probability=True,
       random_state=0,
   )

   clf.fit(X_train, y_train)

   y_prob = clf.predict_proba(X_test)

The adaptive weighting step is handled during ``fit()``, after which prediction
uses the learned composite kernel.