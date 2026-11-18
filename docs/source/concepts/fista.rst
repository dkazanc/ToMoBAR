FISTA regularisation and step selection
=======================================

For FISTA, ``Regularisation.strength`` is the TV weight in the objective
``data_fidelity(image) + strength * TV(image)``. The proximal operator receives
``strength / (S * L)``, where ``S`` is ``num_subsets`` and ``L`` is the explicit
or automatically estimated Lipschitz constant for subset updates. Dividing the
penalty across subsets preserves its coefficient in the full summed objective.
With one subset this is simply ``strength / L``. Ordered-subset FISTA remains
an incremental method, without the convergence guarantee of full-data FISTA.
This scaling is recomputed after geometry changes without modifying the configured strength or the energy calculation.

With ``nonnegativity=True``, the TV proximal solver receives the original
gradient-step image and imposes the constraint inside its updates. Clipping
that image before the TV solve would define a different optimisation problem.
A final projection also keeps the returned image feasible when TV is disabled
or has zero strength. ROF-TV applies nonnegativity after every internal update.

PD-TV uses forward voxel differences with zero outgoing differences and dual
flux at the boundary, matching ``tomobar.energy``. Its primal and extrapolated
images are kept separate, and it returns the constrained primal image. This
corrects previous boundary bias and negative outputs from constrained PD-TV;
regularised reconstruction results can therefore change, including in ADMM.
The corrected fused CUDA step uses one additional full-size primal buffer.
The inner dual step respects the squared-gradient norm bound ``4*ndim`` even
when a smaller bound is supplied. These inner TV steps do not change the outer
FISTA step-selection policy.

Automatic step selection estimates all subset norms and uses their maximum.
PWLS weights are normalised to a maximum of one, so the unweighted norm is
conservative relative to the corresponding weighted norm. Power iteration
itself is an estimate, not a certified upper bound.

This corrects the previous behaviour, which passed the unscaled strength to the
proximal operator. Existing regularised FISTA results and tuned strengths may
therefore change. ADMM's existing ``strength / rho`` scaling is unchanged.
