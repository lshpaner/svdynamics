.. raw:: html

   <div class="no-click">

.. image:: ../assets/svd_logo.svg
   :alt: SVDynamics Logo
   :align: left
   :width: 250px
   :class: dark-light

.. raw:: html

   </div>

Welcome to svdynamics!
======================

**Support Vector Dynamics** (``svdynamics``) is a lightweight Python library
for weighted composite kernels in support vector machines. It supports
fixed-weight kernels for classification and regression and adaptive
kernel-target-alignment weighting for binary classification.

Highlights
----------

- Combine linear, RBF, polynomial, sigmoid, and custom kernels.
- Learn non-negative kernel weights for binary classification.
- Fit normalization scales on training data for consistent predictions.
- Integrate with scikit-learn pipelines, cross-validation, and grid search.
- Inspect decision boundaries, learned weights, and kernel diagnostics.

Explore the :doc:`visualizations` page for reproducible figures or visit the
`interactive SVM Kernel Explorer <https://apps.datasciencedynamics.com/svm_kernel_explorer/>`_.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   getting_started
   visualizations
   api
