Getting Started
===============

TROP: Triply Robust Panel Estimator
-----------------------------------

``trop`` implements the **Triply Robust Panel (TROP)** estimator for average treatment
effects (ATEs) in panel data. The estimator is formulated as a **weighted two-way fixed
effects (TWFE)** objective with distance-based unit/time weights, and optionally includes
a low-rank outcome adjustment via a nuclear-norm penalty.

Reference
^^^^^^^^^
Athey, S., Imbens, G., Qu, Z., Viviano, D. (2025). *Triply Robust Panel Estimators*.
arXiv:2508.21536.

Installation
------------

.. code-block:: bash

   pip install trop

Quickstart
----------

.. code-block:: python

   import numpy as np
   from trop.estimator import TROP_TWFE_average

   # Y: (N, T) outcomes, W: (N, T) treatment indicator
   tau = TROP_TWFE_average(
       Y=Y,
       W=W,
       treated_units=treated_units,
       lambda_unit=0.5,
       lambda_time=0.5,
       lambda_nn=np.inf,   # set finite value to enable low-rank adjustment
       treated_periods=10,
   )

   print("Estimated tau:", tau)

Tuning (placebo cross-validation)
---------------------------------

.. code-block:: python

   from trop.cv import TROP_cv_joint

   best = TROP_cv_joint(
    Y_control=Y_control,
    treated_periods=treated_periods,
    unit_grid=unit_grid,
    time_grid=time_grid,
    nn_grid=nn_grid,
    cv_sampling_method="resample",
    n_trials=200,
    n_treated_units=n_treated_units,
)

   print("Selected (lambda_unit, lambda_time, lambda_nn):", best)

Next steps
----------

See :doc:`api` for the full API reference.
