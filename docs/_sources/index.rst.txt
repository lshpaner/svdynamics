.. raw:: html

   <div class="no-click">

.. image:: ../assets/svd_logo.svg
   :alt: SVDynamics Logo
   :align: left
   :width: 250px

.. raw:: html

   </div>



Welcome to ``svdynamics``! Support Vector Dynamics is a lightweight, scikit-learn 
compatible Python library for building and using mixed (composite) kernels for 
support vector machines. It provides a simple and extensible interface for 
combining multiple kernel functions into a single weighted kernel, while 
remaining fully compatible with existing sklearn pipelines, cross-validation, 
and calibration workflows.

``svdynamics`` focuses on making kernel composition a first-class modeling primitive 
for both classification and regression, without requiring any changes to the 
underlying scikit-learn API.

.. raw:: html
   
   <div style="height: 100px;"></div>


Highlights
----------

- Additive (weighted) composite kernels
- Drop-in replacement for sklearn SVC / SVR
- Compatible with pipelines, GridSearchCV, calibration and resampling
- Designed to integrate cleanly with existing ML workflows

Contents
--------

.. toctree::
   :maxdepth: 2

   getting_started
   examples
   api
