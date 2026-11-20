Configure detector padding and output cropping
==============================================

Use detector padding to extend horizontal edges, and a larger working grid with
an output crop to reduce boundary artefacts. With ``scan`` and ``projections``
from :doc:`configure_geometry` and :doc:`prepare_projections`::

    from tomobar.geometry import Volume3DGeometry
    from tomobar.reconstruction.iterative import IterativeReconstructor

    grid = Volume3DGeometry(shape=(64, 288, 288))
    with IterativeReconstructor(
        projection_geometry=scan,
        volume_geometry=grid,
        detector_padding=16,
        output_shape=(64, 256, 256),
    ) as rec:
        image = rec.CGLS(projections, iterations=15)
        # image.shape == (64, 256, 256)

``detector_padding`` adds pixels per side without changing the supplied scan or
volume. It is also available for direct reconstruction. With padding enabled,
public forward/backprojection operations are not an exact adjoint pair.

``output_shape`` selects a central crop of iterative output; masks are applied
before cropping. Initial images, projection operations and solver state still
use the full working grid. Padding and output cropping are independent.

See :class:`~tomobar.reconstruction.iterative.IterativeReconstructor` for shape
constraints and :doc:`../developers/execution` for allocation details.
