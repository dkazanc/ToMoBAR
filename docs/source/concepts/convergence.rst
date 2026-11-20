Understanding convergence
=========================

Iterations and subsets
-----------------------

A solver iteration updates the reconstruction; ``Regularisation.iterations``
controls the inner penalty solve during that update. Increasing either adds
work, so compare image quality and elapsed time rather than iteration counts
alone. Defaults are starting points for your data.

FISTA and OSEM can visit interleaved subsets within each outer iteration.
Subsets can improve early progress, but ordered-subset FISTA does not inherit
the convergence guarantee of full-data FISTA. ADMM always uses all data;
Landweber, SIRT and CGLS require one subset. See :doc:`../howto/ordered_subsets`.

Strength and step selection
---------------------------

Regularisation strength changes the objective and the balance between image
structure and data fit. The solver rescales it internally for its updates;
use the public strength to compare settings. Method-specific fields such as
TGV's alpha weights have separate roles; see :class:`~tomobar.options.Regularisation`.

FISTA and linearised ADMM can estimate their own step bounds. Power iteration
is an estimate, not a certified upper bound. Before supplying custom constants
or selecting ADMM's CG data solve, consult :doc:`../reference/solver_conventions`.

Energy and stopping
--------------------

An energy history helps reveal stalls or unstable updates, but need not decrease
monotonically. It describes the full working-grid image before output masking
and cropping. Sampling adds projection work; use :class:`~tomobar.options.Diagnostics`
to control its frequency. See :doc:`../howto/monitor_energy`.

ADMM can stop using residual tolerances; the accuracy of its inner data and
regulariser solves can limit attainable tolerances. A small alignment shift
update measures stability of that update, rather than absolute alignment
accuracy. Inspect images and task-specific diagnostics as well as stopping flags.

Use :doc:`../howto/continue_solver` to run batches without losing solver state.
Changing geometry preserves the working image but restarts geometry-dependent
variables; energy values then describe different objectives. Recheck custom
step sizes or Lipschitz constants after a geometry change.
