"""Fixed-geometry adapter for the matched USFFT projector pair."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tomobar.projectors.interface import (
    BackendName,
    ProjectionArray,
    ProjectionGeometry,
    VolumeGeometry,
)

if TYPE_CHECKING:
    from tomobar.projectors._usfft.backend import USFFTBackend

from tomobar.geometry import (
    Parallel2DGeometry,
    Parallel3DGeometry,
    Volume2DGeometry,
    Volume3DGeometry,
    interleaved_view_indices,
)
from tomobar._arrays.prepare import validate_array
from tomobar.projectors.fourier_geometry import require_fourier_geometry
from tomobar._arrays.device import _validate_device_index


class USFFTGeometryBackend:
    """Bind fixed 2D/3D geometry to lazy USFFT numerical resources.

    Args:
        projection_geometry: Angle-based parallel scan in unit detector pixels.
            Scalar centre_of_rotation corrections, including fractional values,
            are supported. Nonzero detector_shifts, vectors and laminography
            require ASTRA and are rejected before numerical resources exist.
        volume_geometry: Matching 2D/3D unit-spaced grid with square slices.
            In 3D, the slice count must equal the detector row count. Detector
            width must be at least the square slice width.
        device_index: Nonnegative CUDA device index, default 0.
        subset_count: Number of nonempty interleaved acquisition-order subsets.

    Notes:
        Volume axes are (Y, X) / (Z, Y, X); projection axes are (angles, detX) /
        (detY, angles, detX). The adapter adds/removes a single slice axis for
        2D execution. NumPy/CuPy inputs retain their library and are packed as
        float32 on the selected GPU. Calls preserve the input.

        The operator includes a circular mask in both directions, unit-pixel
        line-integral scaling and no angular normalisation. Backprojection is
        the discrete transpose up to float32 roundoff. Extra detector bins are
        zero in forward execution and centrally cropped in backprojection.
        The numerical Gaussian interpolation target is 1e-4, not a bound on
        total projection error. Atomic scatter is not bitwise deterministic.

        Numerical resources are created on first use. close() releases owned
        arrays but leaves shared CUDA/CuPy caches intact; repeated cleanup is
        safe. Calls after close() fail. Concurrent execution or cleanup of the
        same instance is unsupported. Optional out is validated and returned
        by identity. GPU execution writes its final result directly into out;
        NumPy out receives a transfer from reusable GPU staging storage. FFT
        internals may still allocate temporary arrays.
    """

    backend_name: BackendName = "usfft"

    def __init__(
        self,
        *,
        projection_geometry: ProjectionGeometry,
        volume_geometry: VolumeGeometry,
        device_index: int = 0,
        subset_count: int = 1,
    ) -> None:
        if not isinstance(
            projection_geometry, (Parallel2DGeometry, Parallel3DGeometry)
        ):
            raise TypeError("USFFT requires a parallel tomography geometry")
        volume_type = (
            Volume3DGeometry
            if isinstance(projection_geometry, Parallel3DGeometry)
            else Volume2DGeometry
        )
        if not isinstance(volume_geometry, volume_type):
            raise TypeError(f"USFFT volume_geometry must be a {volume_type.__name__}")
        require_fourier_geometry(projection_geometry, method="USFFT")
        if isinstance(projection_geometry, Parallel3DGeometry):
            if volume_geometry.shape[0] != projection_geometry.detector_shape[0]:
                raise ValueError("USFFT requires one volume slice per detector row")
        y, x = volume_geometry.shape[-2:]
        if y != x:
            raise ValueError("USFFT requires square volume slices")
        if projection_geometry.detector_shape[-1] < x:
            raise ValueError(
                "USFFT detector width must be at least the volume slice width"
            )
        self.projection_geometry = projection_geometry
        self.volume_geometry = volume_geometry
        self.device_index = _validate_device_index(device_index)
        self.subset_indices = interleaved_view_indices(
            len(projection_geometry.angles), subset_count
        )
        self._numerical: USFFTBackend | None = None
        self._closed = False

    @property
    def subset_count(self) -> int:
        """Number of configured nonempty view subsets."""
        return len(self.subset_indices)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("USFFT backend is closed")

    def _project(
        self,
        data: ProjectionArray,
        subset_index: int | None,
        *,
        forward: bool,
        out: ProjectionArray | None,
    ) -> ProjectionArray:
        self._ensure_open()
        import cupy as cp
        from tomobar.projectors._usfft.backend import USFFTBackend

        projection_shape = list(self.projection_geometry.projection_shape)
        if subset_index is not None:
            if isinstance(subset_index, (bool, np.bool_)) or not isinstance(
                subset_index, (int, np.integer)
            ):
                raise TypeError("subset_index must be an integer")
            if not 0 <= subset_index < self.subset_count:
                raise ValueError("subset_index is out of range")
            projection_shape[-2] = len(self.subset_indices[subset_index])
        domain = self.volume_geometry.shape
        input_shape, output_shape = (
            (domain, projection_shape) if forward else (projection_shape, domain)
        )
        validate_array(
            data, shape=input_shape, device_index=self.device_index, array_module=cp
        )
        if out is not None:
            host = isinstance(data, np.ndarray)
            if isinstance(out, np.ndarray) != host:
                raise TypeError("out must use the same array library as the input")
            validate_array(
                out,
                shape=output_shape,
                device_index=self.device_index,
                array_module=cp,
                name="out",
            )
            if out.dtype != np.float32 or not out.flags.c_contiguous:
                raise ValueError("out must be C-contiguous float32 storage")
            if host and not out.flags.writeable:
                raise ValueError("out must be writable")
            overlaps = np.shares_memory if host else cp.may_share_memory
            if overlaps(data, out):
                raise ValueError("out must not overlap the input")
        if self._numerical is None:
            self._numerical = USFFTBackend(
                size=domain[-1],
                detector_width=self.projection_geometry.detector_shape[-1],
                angles=self.projection_geometry.angles,
                centre_offset=self.projection_geometry.centre_of_rotation,
                subset_count=self.subset_count,
                device_index=self.device_index,
            )
        two_dimensional = len(domain) == 2
        prepared = data[None, ...] if two_dimensional else data
        operation = self._numerical.forward if forward else self._numerical.backproject
        target = out[None, ...] if two_dimensional and out is not None else out
        result = operation(prepared, subset_index=subset_index, out=target)
        if out is not None:
            return out
        return result[0] if two_dimensional else result

    def forward(
        self,
        volume: ProjectionArray,
        *,
        subset_index: int | None = None,
        out: ProjectionArray | None = None,
    ) -> ProjectionArray:
        """Project full data or one subset, preserving library/device and input.

        out must be independent, writable contiguous float32 storage with the
        exact selected projection shape and the input's array library/device.
        """
        return self._project(volume, subset_index, forward=True, out=out)

    def backproject(
        self,
        projections: ProjectionArray,
        *,
        subset_index: int | None = None,
        out: ProjectionArray | None = None,
    ) -> ProjectionArray:
        """Apply the matched transpose to full data or selected subset views.

        Supply only the selected views in acquisition order. out follows the
        forward storage contract, with the full volume geometry's shape.
        """
        return self._project(projections, subset_index, forward=False, out=out)

    def close(self) -> None:
        """Disable execution and release numerical ownership; safe to repeat."""
        self._closed = True
        if self._numerical is not None:
            self._numerical.close()
            self._numerical = None
