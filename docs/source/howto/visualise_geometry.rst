Visualise acquisition geometry
==============================

The code snippets use a projection geometry ``scan``, a working-volume geometry ``grid`` and projection data ``projections``. See :doc:`configure_geometry` for geometry construction and :doc:`../concepts/array_axes` for array layout.

``plot_geometry`` provides a static 3D view of the volume, detector reference
planes and central ray directions. It requires the optional Matplotlib extra
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
Reference planes can intersect the volume in valid parallel geometry; the
plot does not introduce artificial source distances. Arrow lengths are for
display, and a rotation axis is not inferred from arbitrary vectors.

The :doc:`geometry visualisation notebook <../tutorials/geometry_visualisation>`
compares standard tomography, laminography, detector shifts and non-unit
detector pixels. It uses synthetic geometry and can run without a GPU.
