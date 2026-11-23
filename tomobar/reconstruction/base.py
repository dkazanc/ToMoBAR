"""Geometry and resource ownership shared by public reconstruction classes."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING
import weakref
from tomobar._arrays.device import _validate_device_index
from tomobar._arrays.prepare import prepare_projection
import numpy as np

from tomobar.geometry import (
    Parallel2DGeometry,
    Parallel3DGeometry,
    Volume2DGeometry,
    Volume3DGeometry,
)

if TYPE_CHECKING:
    from tomobar.projectors.operator import ProjectionOperator
    from tomobar.projectors.interface import ProjectionBackend


class GeometryReconstruction:
    """Keep acquisition geometry separate from optional detector padding."""

    _projection_backend: ProjectionBackend

    @property
    def Atools(self) -> ProjectionBackend:
        """Read-only legacy alias; use operator for projection calls.

        Retained for existing ASTRA direct-method integrations. This borrows the
        backend; the reconstructor owns cleanup. Use replace_geometry() for
        managed backend replacement and query this property again afterwards.
        """
        return self._projection_backend

    def _init_geometry(self, projection_geometry, volume_geometry, detector_padding):
        if isinstance(projection_geometry, Parallel3DGeometry):
            expected = Volume3DGeometry
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
        self._geometry_version = 0
        self._operator = None
        self._operator_finalizer = None

    def _init_device(self, device_index):
        self._device_index = _validate_device_index(device_index)

    @property
    def device_index(self) -> int:
        """CUDA device selected for this reconstruction."""
        return self._device_index

    @property
    def projection_geometry(self):
        """Unpadded acquisition geometry supplied by the caller."""
        return self._projection_geometry

    @property
    def volume_geometry(self):
        """Working-grid volume geometry."""
        return self._volume_geometry

    @property
    def padded_projection_geometry(self):
        """Acquisition geometry including reconstruction detector padding."""
        return self._padded_projection_geometry

    @property
    def detector_padding(self):
        """Number of extra detector columns on each side."""
        return self._detector_padding

    def _init_operator(self, operator_type, *, subset_count: int = 1) -> None:
        """Own cleanup even when callers retain the borrowed operator."""
        self._operator = operator_type._from_backend(
            self._projection_backend,
            projection_geometry=self.padded_projection_geometry,
            volume_geometry=self.volume_geometry,
            device_index=self.device_index,
            subset_count=subset_count,
        )
        self._operator_finalizer = weakref.finalize(self, self._operator.close)

    @property
    def operator(self) -> ProjectionOperator:
        """Borrow the current fixed operator for the internal padded geometry.

        This reconstructor owns its lifetime. Use the reconstructor's context
        manager or close(), rather than closing the borrowed operator. Owner
        cleanup also closes it if the owner is garbage-collected. Geometry
        replacement closes the previous operator and installs a new one.
        Query this property again after replacement. No projection arrays or
        additional projector handles are allocated by accessing this property.
        """
        self._ensure_open()
        return self._operator

    @property
    def geometry_version(self) -> int:
        """Number of successful geometry replacements (initial geometry is zero)."""
        return self._geometry_version

    def replace_geometry(
        self, projection_geometry: Parallel2DGeometry | Parallel3DGeometry
    ) -> None:
        """Replace acquisition geometry between reconstruction calls.

        Args:
            projection_geometry (Parallel2DGeometry or Parallel3DGeometry):
                Required replacement scan of the same type, detector shape and
                view count as the current geometry. Views retain their acquisition
                order. Passing the current geometry object is a no-op.

        Returns:
            None: Updates this reconstructor's geometry and increments
            geometry_version after a successful replacement.

        The geometry type, detector dimensions and view count must stay fixed;
        data must retain the same acquisition ordering. Volume grid, device,
        padding, output crop and ordered-subset configuration are preserved.
        An identical geometry object is a no-op.

        Build and validate the replacement before releasing the old wrapper.
        All old full-view/subset projectors are then released, and replacement
        handles are created lazily on the next projection. Validation or wrapper
        construction failure leaves the current geometry and handles unchanged.
        Actual CUDA projector construction still occurs on first use.

        Solver normalisation, sensitivity and automatic Lipschitz estimates
        are local to each call and are recomputed on the next reconstruction.
        Caller-supplied step sizes or Lipschitz constants must be reconsidered
        after a geometry change. No reconstruction images or measured data are
        owned or modified here; pass an existing full-grid image via ``initial``
        to warm-start the next solver call. This method must not run concurrently
        with projection or reconstruction on the same object.
        """
        self._ensure_open()
        current = self.projection_geometry
        if type(projection_geometry) is not type(current):
            raise TypeError(f"projection_geometry must be a {type(current).__name__}")
        if projection_geometry.projection_shape != current.projection_shape:
            raise ValueError(
                "replacement geometry must preserve detector shape and view count"
            )
        if projection_geometry is current:
            return
        shape = projection_geometry.detector_shape
        padded = (
            replace(
                projection_geometry,
                detector_shape=(*shape[:-1], shape[-1] + 2 * self.detector_padding),
            )
            if self.detector_padding
            else projection_geometry
        )
        previous = self._projection_backend
        previous_operator = self.operator
        previous_finalizer = self._operator_finalizer
        replacement = previous_operator._create_backend(
            backend=previous_operator.backend, projection_geometry=padded
        )
        try:
            replacement_operator = (
                type(self._operator)._from_backend(
                    replacement,
                    projection_geometry=padded,
                    volume_geometry=self.volume_geometry,
                    device_index=self.device_index,
                    subset_count=self._operator.subset_count,
                )
                if self._operator is not None
                else None
            )
            # Keep user-selected 2D FBP filter settings across wrapper replacement.
            for name in ("fbp_filter_type", "fbp_filter_parameter", "fbp_filter_d"):
                if hasattr(previous, name):
                    setattr(replacement, name, getattr(previous, name))
            replacement_finalizer = (
                weakref.finalize(self, replacement_operator.close)
                if replacement_operator is not None
                else None
            )
        except Exception:
            replacement.close()
            raise
        self._projection_geometry = projection_geometry
        self._padded_projection_geometry = padded
        self._projection_backend = replacement
        self._operator = replacement_operator
        self._operator_finalizer = replacement_finalizer
        self._geometry_version += 1
        if previous_finalizer is not None:
            previous_finalizer()
        elif previous_operator is not None:
            previous_operator.close()
        else:
            previous.close()

    def _prepare_projection(
        self,
        data,
        *,
        array_module,
        input_axes=None,
        subset_index=None,
        gpu=False,
        pack=False,
        allow_single_row=False,
        target_axes=None,
        pad_detector=True,
        name="data",
        own=False,
    ):
        """Prepare acquired columns for one projection consumer."""
        self._ensure_open()
        shape = self.projection_geometry.projection_shape
        if subset_index is not None:
            if isinstance(subset_index, (bool, np.bool_)) or not isinstance(
                subset_index, (int, np.integer)
            ):
                raise TypeError("subset_index must be an integer or None")
            if not 0 <= subset_index < len(self.subset_indices):
                raise ValueError("subset_index is out of range")
            shape = (shape[0], len(self.subset_indices[subset_index]), shape[2])
        return prepare_projection(
            data,
            shape=shape,
            device_index=self.device_index,
            array_module=array_module,
            input_axes=input_axes,
            padding=self.detector_padding if pad_detector else 0,
            gpu=gpu,
            pack=pack,
            allow_single_row=allow_single_row,
            target_axes=target_axes,
            name=name,
            own=own,
        )

    def _crop_projection(self, data):
        padding = self.detector_padding
        return data[..., padding:-padding] if padding else data

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("The reconstruction object has been closed")

    def close(self):
        """Release all lazily created projector handles. Repeated calls are safe."""
        self._closed = True
        finalizer = getattr(self, "_operator_finalizer", None)
        if finalizer is not None:
            finalizer()
        elif hasattr(self, "_projection_backend"):
            self._projection_backend.close()

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, *exc):
        self.close()
