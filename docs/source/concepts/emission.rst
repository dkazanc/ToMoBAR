Emission reconstruction and sensitivity
=======================================

``OSEM`` performs MLEM when ``num_subsets`` is 1. Each unregularized subset update
uses the current subset's sensitivity as a denominator:

.. code-block:: text

    sensitivity = A_s.T(ones)
    x *= A_s.T(data_s / max(A_s(x), epsilon)) / max(sensitivity, epsilon)

The previous implementation multiplied by sensitivity and reused subset zero's
sensitivity. Both errors are corrected. MLEM/OSEM intensity scaling and existing
numerical reference values therefore change. Regularisation strengths tuned to
the previous scaling may need adjustment.

Sensitivity is guarded at ``epsilon=1e-8``. A voxel with zero sensitivity has a
zero backprojection and receives zero multiplicative correction, avoiding
NaNs or infinities. MLEM computes its single sensitivity once per call. OSEM
recomputes sensitivity for the current subset at each update and releases it
before regularisation, trading an extra subset backprojection per update for
GPU storage bounded by one sensitivity volume instead of ``num_subsets`` volumes.
Projector handles remain cached and are reused for these extra backprojections.
