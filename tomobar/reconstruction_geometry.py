"""Geometry and resource ownership shared by public reconstruction classes."""

from dataclasses import replace
from tomobar.supp.device import _validate_device_index
import numpy as np

from tomobar.geometry import (
    Parallel2DGeometry,
    Parallel3DGeometry,
    Volume2DGeometry,
    VolumeGeometry,
)


class GeometryReconstruction:
    """Keep acquisition geometry separate from optional detector padding."""

    def _init_geometry(self, projection_geometry, volume_geometry, detector_padding):
        if isinstance(projection_geometry, Parallel3DGeometry):
            expected = VolumeGeometry
            self.geom = "3D"
        elif isinstance(projection_geometry, Parallel2DGeometry):
            expected = Volume2DGeometry
            self.geom = "2D"
        else:
            raise TypeError(
                "projection_geometry must be a Parallel2DGeometry or Parallel3DGeometry"
            )
        if not isinstance(volume_geometry, expected):
            raise TypeError(f"volume_geometry must be a {expected.__name__}")
        if isinstance(detector_padding, (bool, np.bool_)) or not isinstance(
            detector_padding, (int, np.integer)
        ):
            raise TypeError("detector_padding must be an integer")
        if detector_padding < 0:
            raise ValueError("detector_padding cannot be negative")
        self._projection_geometry = projection_geometry
        self._volume_geometry = volume_geometry
        self._detector_padding = int(detector_padding)
        shape = projection_geometry.detector_shape
        self._padded_projection_geometry = (
            replace(
                projection_geometry,
                detector_shape=(*shape[:-1], shape[-1] + 2 * detector_padding),
            )
            if detector_padding
            else projection_geometry
        )
        self._closed = False

    def _init_device(self, device_index):
        self._device_index = _validate_device_index(device_index)

    @property
    def device_index(self) -> int:
        return self._device_index

    @property
    def projection_geometry(self):
        return self._projection_geometry

    @property
    def volume_geometry(self):
        return self._volume_geometry

    @property
    def padded_projection_geometry(self):
        return self._padded_projection_geometry

    @property
    def detector_padding(self):
        return self._detector_padding

    def _crop_projection(self, data):
        padding = self.detector_padding
        return data[..., padding:-padding] if padding else data

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("The reconstruction object has been closed")

    def close(self):
        """Release all lazily created projector handles. Repeated calls are safe."""
        self._closed = True
        if hasattr(self, "Atools"):
            self.Atools.close()

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, *exc):
        self.close()
