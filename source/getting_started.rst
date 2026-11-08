Getting started
===============

Installation
------------

Install ``svdynamics`` from PyPI:

.. code-block:: bash

   pip install svdynamics

For this prerelease specifically:

.. code-block:: bash

   pip install svdynamics==0.0.1a2


Basic usage
-----------

The central concept in ``svdynamics`` is the ``CompositeKernel`` object.
A composite kernel combines multiple base kernels in a weighted sum while
retaining the familiar scikit-learn SVM interface.

.. code-block:: python

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
   y_pred = clf.predict(X)
   y_prob = clf.predict_proba(X)

The estimator fits its own copy of the supplied kernel. The original
``kernel`` configuration remains unchanged; fitted state is available
through ``clf.kernel_``.


How composite kernels work
--------------------------

Composite kernels are implemented as callable kernels supplied to
scikit-learn's SVC or SVR. A composite kernel is a weighted sum of base
kernels:

.. math::

   K(x, x') = \sum_{i=1}^{m} w_i \, K_i(x, x')

Here, :math:`K_i` is the i-th base kernel and :math:`w_i` is its weight.
With ``CompositeKernel``, users specify the weights directly, for example
``weights=[0.6, 0.3, 0.1]``. Weights must be nonnegative and cannot all be
zero; zero can be used to disable a component.

When normalization is enabled, the base kernels are scaled before their
weighted sum is computed.


Kernel normalization
--------------------

With ``normalize=True``, each base kernel is scaled to reduce differences
in magnitude between kernel matrices. In ``0.0.1a1``, the scales are
computed from training data and reused for predictions. This prevents a
point's decision score from changing merely because it was evaluated in a
different prediction batch.

``SVDClassifier`` and ``SVDRegressor`` fit normalization scales as part of
estimator fitting. Inspect the fitted scales through the estimator:

.. code-block:: python

   print(clf.kernel_.scales_)

When using a ``CompositeKernel`` directly, fit its scales explicitly on
training observations before evaluating new data:

.. code-block:: python

   kernel.fit_scales(X)
   print(kernel.scales_)

Evaluating a normalized kernel on new data without fitted scales emits a
warning.


Adaptive composite kernels
--------------------------

``AdaptiveCompositeKernel`` learns weights from training data rather than
requiring them to be specified manually. It measures the alignment of each
candidate kernel matrix with the binary training targets and derives
nonnegative, normalized weights.

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

   clf = SVDClassifier(C=1.0, kernel=kernel)
   clf.fit(X_train, y_train)

   y_pred = clf.predict(X_test)

   print(clf.kernel_.alignments_)
   print(clf.kernel_.weights_)

The estimator's ``kernel_`` contains the learned state. The original
``kernel`` passed to the estimator is not fitted in place. Calling
``AdaptiveCompositeKernel.fit(X_train, y_train)`` directly remains supported.


Kernel-target alignment
-----------------------

For a kernel matrix :math:`K` and target kernel :math:`K_y`, the uncentered
kernel-target alignment is

.. math::

   A(K, K_y) =
   \frac{\langle K, K_y \rangle_F}
   {\|K\|_F \, \|K_y\|_F}

where :math:`\langle \cdot, \cdot \rangle_F` denotes the Frobenius inner
product and :math:`\|\cdot\|_F` the Frobenius norm.

For binary classification, the target kernel is

.. math::

   K_y = \tilde{y}\tilde{y}^{T}

where :math:`\tilde{y}` contains the class labels mapped internally to
-1 and +1. This makes the learned alignment scores invariant to the
supported binary-label encodings, including 0/1, -1/+1, and strings.

Negative alignment scores are clipped to zero, and the remaining scores
are normalized into weights. If none are positive, uniform weights are
used. Larger positive alignments receive greater weight.


Learned kernel weights
----------------------

For the adaptive classifier fitted above, inspect the learned quantities
through ``clf.kernel_``:

.. code-block:: python

   print(clf.kernel_.alignments_)
   print(clf.kernel_.weights_)

``alignments_`` stores the candidate kernel-target alignment scores, and
``weights_`` stores the resulting nonnegative, normalized kernel weights.
For illustration, a fitted model might produce:

.. code-block:: text

   alignments_: [0.0027, 0.1476, 0.0206]
   weights_:    [0.0160, 0.8635, 0.1205]

These values are illustrative, not guaranteed output for the example above.
The RBF kernel receives the largest weight in this illustration because
its alignment is highest.

The resulting adaptive composite kernel can be written as

.. math::

   K_{\mathrm{adaptive}}(x, x') =
   \sum_{i=1}^{m} \hat{w}_i K_i(x, x')

where :math:`\hat{w}_i` is a learned weight. When normalization is enabled,
the base kernels in this expression are understood to be normalized using
the training-derived scales.


Fixed versus adaptive weighting
-------------------------------

``CompositeKernel`` uses weights specified by the user.
``AdaptiveCompositeKernel`` learns weights from binary training targets
using uncentered kernel-target alignment.

Adaptive weighting is a training-data heuristic, not direct optimization
of held-out predictive performance. It does not guarantee an improvement
over a single RBF kernel or another fixed kernel configuration.


Adaptive weighting requirements
-------------------------------

``AdaptiveCompositeKernel`` supports binary classification only. It
accepts supported two-class encodings such as 0/1, -1/+1, and strings,
and maps the two classes internally to -1 and +1. Multiclass targets
raise ``ValueError``.

``SVDRegressor`` supports fixed ``CompositeKernel`` configurations, but
not ``AdaptiveCompositeKernel``. Passing an adaptive kernel to the
regressor raises ``TypeError``.


Using the fitted classifier
---------------------------

Once fitted, ``SVDClassifier`` exposes the familiar scikit-learn
prediction interface:

.. code-block:: python

   y_pred = clf.predict(X_test)
   scores = clf.decision_function(X_test)

For probability estimates, enable probability estimation before fitting:

.. code-block:: python

   clf = SVDClassifier(
       C=1.0,
       kernel=kernel,
       probability=True,
       random_state=0,
   )
   clf.fit(X_train, y_train)
   y_prob = clf.predict_proba(X_test)

The fitted adaptive weights and normalization scales are reused at
prediction time.


Kernel validation and diagnostics
---------------------------------

``svdynamics`` validates kernel-matrix dimensions, finite values, and
symmetry of training Gram matrices. Invalid custom kernel outputs raise
descriptive errors.

Inspect the smallest Gram-matrix eigenvalue of each base kernel using
``min_eigenvalues``:

.. code-block:: python

   eigenvalues = clf.kernel_.min_eigenvalues(X_train)
   print(eigenvalues)

Custom and sigmoid kernels are not guaranteed to be positive semidefinite.
This diagnostic can help identify indefinite kernels, but does not modify
or repair them.


Package help
------------

Display the package overview without changing Python's built-in
``help()`` function:

.. code-block:: python

   import svdynamics

   svdynamics.help()