Visualise acquisition geometry
==============================

Install ``tomobar[visualisation]`` and use ``scan`` / ``grid`` from
:doc:`configure_geometry`. Plotting requires Matplotlib, but no ASTRA or CUDA::

    from tomobar.visualisation import plot_geometry

    fig, ax = plot_geometry(scan, grid)
    fig.savefig("geometry.png", dpi=200)

The default view shows the fixed volume, beam directions and detectors at
schematic downstream distances. Known tomography and laminography trajectories
also show a rotation axis and acquisition-rotation arc. The display does not
change the scan geometry.

Use ``view_indices=[0, 30, 60]`` to select acquisition indices, or
``detector_position="reference"`` to inspect the original detector planes.
To keep the instrument fixed and show sample rotation instead::

    fig, ax = plot_geometry(scan, grid, view_indices=[0], frame="laboratory")

Laboratory mode requires a known tomography or untransformed laminography
trajectory. Arbitrary vectors have no inferred rotation axis; stationary or
reversing scans have no direction arc.

See :func:`~tomobar.visualisation.plot_geometry` for display options and the
:doc:`geometry notebook <../tutorials/geometry_visualisation>` for comparisons.
