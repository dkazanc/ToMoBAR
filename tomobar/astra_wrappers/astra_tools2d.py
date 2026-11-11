from tomobar.supp.device import _validate_device_index
from contextlib import ExitStack
import astra
from tomobar.astra_wrappers.astra_base import AstraBase
from tomobar.astra_wrappers.geometry import (
    parallel2d_vectors,
    projection_geometry as to_astra_projection_geometry,
    volume_geometry as to_astra_volume_geometry,
)
from tomobar.geometry import (
    Parallel2DGeometry,
    Volume2DGeometry,
    _validate_ordsub_number,
)

import numpy as np


###########Child class############
class AstraTools2D(AstraBase):
    """Bind explicit 2D scan and volume geometry to ASTRA GPU projectors.

    Detector padding is already represented in projection_geometry. Generate
    full-view vectors once; ordered subsets select rows from those vectors.
    Forward/backprojection use ASTRA 2.5's direct NumPy/CuPy array API. Handles
    for all views and each used subset are created lazily and shared until
    close(). Filtered 2D FBP keeps a separate algorithm path.
    """

    def __init__(
        self,
        *,
        projection_geometry: Parallel2DGeometry,
        volume_geometry: Volume2DGeometry,
        device_index: int = 0,
        ordsub_number: int | None = None,
        verbosity: bool = False,
    ):
        if not isinstance(projection_geometry, Parallel2DGeometry):
            raise TypeError("projection_geometry must be a Parallel2DGeometry")
        if not isinstance(volume_geometry, Volume2DGeometry):
            raise TypeError("volume_geometry must be a Volume2DGeometry")
        self._projection_geometry = projection_geometry
        self._volume_geometry = volume_geometry
        self._device_index = _validate_device_index(device_index)
        self._ordsub_number = _validate_ordsub_number(
            ordsub_number, projection_geometry.angles.size
        )
        self._setOS_indices(projection_geometry.angles.size)
        self._init_projector_resources()
        self.fbp_filter_type = "ram-lak"
        self.fbp_filter_parameter = None
        self.fbp_filter_d = None
        self.vol_geom = to_astra_volume_geometry(volume_geometry)
        self.vectors = parallel2d_vectors(projection_geometry)
        self.proj_geom = to_astra_projection_geometry(projection_geometry, self.vectors)
        self.proj_geom_OS = {
            index: to_astra_projection_geometry(
                projection_geometry, self.vectors, indices
            )
            for index, indices in enumerate(self.subset_indices)
        }
        if verbosity:
            print("2D geometry initialised; projectors will be created on first use...")

    @property
    def projection_geometry(self):
        return self._projection_geometry

    @property
    def volume_geometry(self):
        return self._volume_geometry

    @property
    def device_index(self):
        return self._device_index

    @property
    def ordsub_number(self):
        return self._ordsub_number

    def _fbp(self, sinogram: np.ndarray) -> np.ndarray:
        """Run ASTRA's filtered 2D reconstruction with scoped temporary data."""
        self._ensure_open()
        with ExitStack() as resources:
            sinogram_id = astra.data2d.create("-sino", self.proj_geom, sinogram)
            resources.callback(astra.data2d.delete, sinogram_id)
            rec_id = astra.data2d.create("-vol", self.vol_geom)
            resources.callback(astra.data2d.delete, rec_id)
            cfg = astra.astra_dict("FBP_CUDA")
            cfg["option"] = {"GPUindex": self.device_index}
            cfg["ReconstructionDataId"] = rec_id
            cfg["ProjectionDataId"] = sinogram_id
            cfg["FilterType"] = self.fbp_filter_type
            if self.fbp_filter_parameter is not None:
                cfg["FilterParameter"] = self.fbp_filter_parameter
            if self.fbp_filter_d is not None:
                cfg["FilterD"] = self.fbp_filter_d
            alg_id = astra.algorithm.create(cfg)
            resources.callback(astra.algorithm.delete, alg_id)
            astra.algorithm.run(alg_id, 1)
            return astra.data2d.get(rec_id)
