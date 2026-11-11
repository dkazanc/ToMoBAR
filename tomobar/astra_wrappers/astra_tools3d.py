from tomobar.astra_wrappers.astra_base import AstraBase
from tomobar.astra_wrappers.geometry import (
    parallel3d_vectors,
    projection_geometry as to_astra_projection_geometry,
    volume_geometry as to_astra_volume_geometry,
)
from tomobar.geometry import Parallel3DGeometry, VolumeGeometry

import astra
import numpy as np

try:
    import cupy as xp

    try:
        xp.cuda.Device(0).compute_capability
    except xp.cuda.runtime.CUDARuntimeError:
        import numpy as xp
except ImportError:
    import numpy as xp

from typing import Optional
from tomobar.supp.funcs import _validate_ordsub_number


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
    processing_arch = "gpu"

    def __init__(
        self,
        *,
        projection_geometry: Parallel3DGeometry,
        volume_geometry: VolumeGeometry,
        device_index: int = 0,
        ordsub_number: Optional[int] = None,
        verbosity: bool = False,
    ):
        if not isinstance(projection_geometry, Parallel3DGeometry):
            raise TypeError("projection_geometry must be a Parallel3DGeometry")
        if not isinstance(volume_geometry, VolumeGeometry):
            raise TypeError("volume_geometry must be a VolumeGeometry")
        self._projection_geometry = projection_geometry
        self._volume_geometry = volume_geometry
        self._ordsub_number = _validate_ordsub_number(
            ordsub_number, projection_geometry.angles.size
        )
        self._setOS_indices(projection_geometry.angles.size)
        if isinstance(device_index, (bool, np.bool_)) or not isinstance(
            device_index, (int, np.integer)
        ):
            raise TypeError("device_index must be a nonnegative integer")
        if device_index < 0:
            raise ValueError("device_index must be a nonnegative integer")
        self._device_index = int(device_index)
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
        self._projector_handles = {}
        if verbosity:
            print("3D geometry initialised; projectors will be created on first use...")

    @property
    def projection_geometry(self) -> Parallel3DGeometry:
        return self._projection_geometry

    @property
    def volume_geometry(self) -> VolumeGeometry:
        return self._volume_geometry

    @property
    def ordsub_number(self) -> int:
        return self._ordsub_number

    @property
    def device_index(self):
        return self._device_index

    def _selected_geometry(self, subset_index):
        if subset_index is None:
            return self.proj_geom
        if isinstance(subset_index, (bool, np.bool_)) or not isinstance(
            subset_index, (int, np.integer)
        ):
            raise TypeError("subset_index must be an integer or None")
        if not 0 <= subset_index < self.ordsub_number:
            raise ValueError("subset_index is out of range")
        return self.proj_geom_OS[int(subset_index)]

    def _projector_handle(self, subset_index):
        """Cache one handle for all views and one for each used subset."""
        self._ensure_open()
        geometry = self._selected_geometry(subset_index)
        if subset_index not in self._projector_handles:
            handle = astra.create_projector(
                "cuda3d",
                geometry,
                self.vol_geom,
                options={"GPUindex": self.device_index},
            )
            self._register_projector(handle)
            self._projector_handles[subset_index] = handle
        return self._projector_handles[subset_index]

    def _project(self, data, subset_index, *, forward):
        self._ensure_open()
        projection = self._selected_geometry(subset_index)
        input_geometry = self.vol_geom if forward else projection
        output_geometry = projection if forward else self.vol_geom
        if isinstance(data, np.ndarray):
            if data.ndim != 3 or data.shape != tuple(astra.geom_size(input_geometry)):
                raise ValueError(
                    f"Input must have shape {astra.geom_size(input_geometry)}, got {data.shape}"
                )
            if data.dtype.kind not in "biuf":
                raise TypeError("Input must have a real numeric dtype")
            prepared = np.ascontiguousarray(data, dtype=np.float32)
            out = np.empty(astra.geom_size(output_geometry), dtype=np.float32)
            handle = self._projector_handle(subset_index)
            operation = (
                astra.projector3d.direct_FP if forward else astra.projector3d.direct_BP
            )
            operation(handle, prepared, out=out)
        else:
            prepared = self._prepare_cupy_input(data, input_geometry, "data")
            with xp.cuda.Device(self.device_index):
                out = xp.empty(astra.geom_size(output_geometry), dtype=xp.float32)
                handle = self._projector_handle(subset_index)
                operation = (
                    astra.projector3d.direct_FP
                    if forward
                    else astra.projector3d.direct_BP
                )
                operation(handle, prepared, out=out)
        return out

    def forward(self, volume, *, subset_index=None):
        """Project a NumPy/CuPy volume using ASTRA 2.5's direct DLPack API.

        None selects all views, even when subsets are configured. A subset
        index selects its interleaved views. Output uses the input array library.
        Noncontiguous or non-float32 real inputs are packed as float32.
        """
        return self._project(volume, subset_index, forward=True)

    def backproject(self, projections, *, subset_index=None):
        """Backproject NumPy/CuPy projections using the same cached handle."""
        return self._project(projections, subset_index, forward=False)

    def close(self):
        super().close()
        self._projector_handles.clear()
