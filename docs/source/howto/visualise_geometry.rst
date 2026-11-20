Visualise acquisition geometry
==============================

The code snippets use a projection geometry ``scan``, a working-volume geometry ``grid`` and projection data ``projections``. See :doc:`configure_geometry` for geometry construction and :doc:`../concepts/array_axes` for array layout.

``plot_geometry`` provides a static 3D view of the volume, downstream detector
planes, labelled beam directions and rotation axes. It requires the optional Matplotlib extra
(``pip install 'tomobar[visualisation]'``), but neither ASTRA nor CUDA::

    from tomobar.visualisation import plot_geometry

    fig, ax = plot_geometry(scan, grid, view_indices=[0, 30, 60])
    fig.savefig("geometry.png", dpi=200)

Omit ``view_indices`` to show up to three evenly spaced acquisition indices.
Pass an existing Matplotlib 3D axes as ``ax=...`` to compose a larger figure.
The function returns the figure and axes without calling ``show()``. All axes
use equal spatial scaling in XYZ voxel coordinates; volume array shapes remain
ZYX. Detector boundaries follow outer pixel edges and preserve the supplied
column/row basis lengths and angles. Shifts and CoR are applied through
``to_vectors()`` exactly once.

Tomography, laminography and explicit ``Parallel3DGeometry`` vectors are
supported. The first version accepts ``Volume3DGeometry`` and 3D scans only.
By default, detector planes are moved downstream for display, with a clear gap
from the volume and beam arrows ending at their centres. The distances are
schematic; the scan vectors and projection data are unchanged. Detector shifts,
CoR and pixel bases are preserved. To inspect the supplied reference planes,
which can intersect the volume, use ``detector_position="reference"``.
Arrow lengths are for display. Each selected view labels its beam and detector. Known tomography
and laminography trajectories also show a dashed rotation axis and a curved
arrow for acquisition rotation around the fixed sample. The arrow follows the
full acquisition order, even when only one view is displayed; stationary or
reversing scans omit it. A rotation axis is not inferred from arbitrary vectors.

Use ``frame="laboratory"`` to keep the beam and detector fixed and show the
sample rotating instead::

    fig, ax = plot_geometry(scan, grid, view_indices=[0], frame="laboratory")

In this frame the rotation arc indicates sample motion, opposite to acquisition
rotation in the default sample frame. The detector is placed downstream at a
schematic distance by default, with incoming beam arrows and detector pixel basis labels
``u`` and ``v``. This display does not alter the scan geometry. Laboratory mode
requires a known tomography or untransformed laminography trajectory.

The :doc:`geometry visualisation notebook <../tutorials/geometry_visualisation>`
compares standard tomography, laminography, detector shifts and non-unit
detector pixels. It uses synthetic geometry and can run without a GPU.
