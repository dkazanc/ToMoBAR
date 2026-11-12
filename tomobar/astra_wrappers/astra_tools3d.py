from tomobar.supp.device import _validate_device_index
from tomobar.astra_wrappers.astra_base import AstraBase
from tomobar.astra_wrappers.geometry_adapter import (
    parallel3d_vectors,
    projection_geometry as to_astra_projection_geometry,
    volume_geometry as to_astra_volume_geometry,
)
from tomobar.geometry import (
    Parallel3DGeometry,
    Volume3DGeometry,
    _validate_ordsub_number,
)

import astra
import numpy as np

from typing import Optional


###########Child class############
class AstraTools3D(AstraBase):
    """Bind explicit 3D geometry to ASTRA GPU projection/backprojection.

    The scan and volume objects are the source of geometry and validation.
    Detector padding belongs to the caller: detector_shape must describe the
    projection arrays actually passed to ASTRA, including any padded columns.
    Volume shape is independent of detector shape and may be non-cubic.

    Args:
        projection_geometry: Acquisition angles, detector shape, shifts and CoR.
        volume_geometry: Reconstruction grid in (z, y, x) order.
        device_index: GPU device index, defaulting to zero.
        ordsub_number: Number of interleaved view subsets. None means one.
        verbosity: Print an initialisation message when True.
    """

    _is_3d = True

    def __init__(
        self,
        *,
        projection_geometry: Parallel3DGeometry,
        volume_geometry: Volume3DGeometry,
        device_index: int = 0,
        ordsub_number: Optional[int] = None,
        verbosity: bool = False,
    ):
        if not isinstance(projection_geometry, Parallel3DGeometry):
            raise TypeError("projection_geometry must be a Parallel3DGeometry")
        if not isinstance(volume_geometry, Volume3DGeometry):
            raise TypeError("volume_geometry must be a Volume3DGeometry")
        self._projection_geometry = projection_geometry
        self._volume_geometry = volume_geometry
        self._ordsub_number = _validate_ordsub_number(
            ordsub_number, projection_geometry.view_count
        )
        self._setOS_indices(projection_geometry.view_count)
        self._device_index = _validate_device_index(device_index)
        self._init_projector_resources()

        self.vol_geom = to_astra_volume_geometry(volume_geometry)
        self.vectors = parallel3d_vectors(projection_geometry)
        self.proj_geom = to_astra_projection_geometry(projection_geometry, self.vectors)
        self.proj_geom_OS = {
            index: to_astra_projection_geometry(
                projection_geometry, self.vectors, indices
            )
            for index, indices in enumerate(self.subset_indices)
        }
        if verbosity:
            print("3D geometry initialised; projectors will be created on first use...")

    @property
    def projection_geometry(self) -> Parallel3DGeometry:
        return self._projection_geometry

    @property
    def volume_geometry(self) -> Volume3DGeometry:
        return self._volume_geometry

    @property
    def ordsub_number(self) -> int:
        return self._ordsub_number

    @property
    def device_index(self):
        return self._device_index
