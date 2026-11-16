"""Fixed-geometry projection operators with selectable GPU backends."""

import cupy as cp
import numpy as np

from tomobar._projection_backend import (
    BackendName,
    ProjectionBackend,
    ProjectionGeometry,
    VolumeGeometry,
    create_projection_backend,
)
from tomobar.geometry import (
    Parallel2DGeometry,
    Parallel3DGeometry,
    Volume2DGeometry,
    Volume3DGeometry,
    interleaved_view_indices,
)
from tomobar.supp.device import _on_device, _validate_device_index
from tomobar.options import _positive_integer

__all__ = ["ProjectionOperator"]


class ProjectionOperator:
    """Forward projection and backend-specific backprojection with fixed geometry.

    Keyword Args:
        projection_geometry (Parallel2DGeometry or Parallel3DGeometry): Required
            acquisition geometry. Views retain their acquisition order.
        volume_geometry (Volume2DGeometry or Volume3DGeometry): Required volume
            geometry, matching the dimensionality of the acquisition geometry.
        backend (str): "astra" (default) or "usfft". USFFT requires square
            slices, matching detector rows, detector width at least the slice
            width, and angle geometry. Scalar centre_of_rotation corrections are
            supported; nonzero detector_shifts (constant or per-view), vector
            geometry and laminography require ASTRA.
        subset_count (int): Positive interleaved subset count, default 1.
            Must not exceed the number of acquired views. Each execution call
            defaults to using all views.
        device_index (int): Nonnegative CUDA device index, default 0. Both NumPy
            and CuPy execution require CUDA. NumPy results stay on the host;
            CuPy results stay on the selected GPU.

    Notes:
        Volume axes are (Y, X) in 2D and (Z, Y, X) in 3D. Projection axes are
        (angles, detX) in 2D and (detY, angles, detX) in 3D. Geometry, shapes and
        device selection are read-only. Create another operator to change them.

        With ASTRA, operations use the supplied geometry directly. No detector padding,
        cropping, masks, filtering or angular normalisation is applied.
        Backprojection uses ASTRA's approximate adjoint; an exact transpose
        relationship is not guaranteed.

        USFFT uses a circular support mask in both directions, unit-pixel
        line-integral scaling, and matched gather/scatter weights. Backprojection
        is its Euclidean transpose up to float32 roundoff. Subsets select rows
        of the same full operator; no angular normalisation is applied. Detector
        padding is zero, with central cropping in the transpose operation.

        Projectors are created lazily and reused across calls. Use a context
        manager or close() to release them promptly. The backend also releases
        handles when an abandoned operator is garbage-collected. Concurrent
        calls or cleanup on the same operator are unsupported.
    """

    __slots__ = (
        "_backend",
        "_projection_geometry",
        "_volume_geometry",
        "_device_index",
        "_subset_indices",
        "__weakref__",
    )

    def __init__(
        self,
        *,
        projection_geometry: Parallel2DGeometry | Parallel3DGeometry,
        volume_geometry: Volume2DGeometry | Volume3DGeometry,
        device_index: int = 0,
        subset_count: int = 1,
        backend: BackendName = "astra",
    ) -> None:
        self._set_geometry(
            projection_geometry, volume_geometry, device_index, subset_count
        )
        self._backend: ProjectionBackend = self._create_backend(
            backend=backend, projection_geometry=projection_geometry
        )

    def _create_backend(
        self, *, backend: BackendName, projection_geometry: ProjectionGeometry
    ) -> ProjectionBackend:
        """Build a replacement with the same volume, device and subset policy."""
        return create_projection_backend(
            backend=backend,
            projection_geometry=projection_geometry,
            volume_geometry=self.volume_geometry,
            device_index=self.device_index,
            subset_count=self.subset_count,
        )

    def _set_geometry(
        self,
        projection_geometry: ProjectionGeometry,
        volume_geometry: VolumeGeometry,
        device_index: int,
        subset_count: int,
    ) -> None:
        if not isinstance(
            projection_geometry, (Parallel2DGeometry, Parallel3DGeometry)
        ):
            raise TypeError(
                "projection_geometry must be a Parallel2DGeometry or Parallel3DGeometry"
            )
        self._projection_geometry = projection_geometry
        self._volume_geometry = volume_geometry
        self._device_index = _validate_device_index(device_index)
        self._subset_indices = interleaved_view_indices(
            projection_geometry.projection_shape[-2],
            _positive_integer(subset_count, "subset_count"),
        )

    @classmethod
    def _from_backend(
        cls,
        backend: ProjectionBackend,
        *,
        projection_geometry: ProjectionGeometry,
        volume_geometry: VolumeGeometry,
        device_index: int,
        subset_count: int,
    ) -> "ProjectionOperator":
        """Bind prepared projection operations; the reconstructor owns cleanup."""
        operator = cls.__new__(cls)
        operator._set_geometry(
            projection_geometry, volume_geometry, device_index, subset_count
        )
        operator._backend = backend
        return operator

    @property
    def backend(self) -> BackendName:
        """Selected numerical backend: astra or usfft."""
        return self._backend.backend_name

    @property
    def subset_count(self) -> int:
        """Number of configured interleaved view subsets."""
        return len(self._subset_indices)

    @property
    def subset_indices(self) -> tuple[np.ndarray, ...]:
        """Immutable acquisition-order view indices for each subset."""
        return self._subset_indices

    def subset_shape(self, subset_index: int) -> tuple[int, ...]:
        """Return the projection shape for one configured subset.

        Args:
            subset_index (int): Required zero-based index into subset_indices.
                Negative indices and booleans are rejected.

        Returns:
            tuple[int, ...]: (angles, detX) in 2D or (detY, angles, detX) in 3D,
            with only the selected subset's views. No CUDA work is performed.
            range_shape continues to describe the complete acquisition.
        """
        if isinstance(subset_index, (bool, np.bool_)) or not isinstance(
            subset_index, (int, np.integer)
        ):
            raise TypeError("subset_index must be an integer")
        if not 0 <= subset_index < self.subset_count:
            raise ValueError("subset_index is out of range")
        shape = list(self.range_shape)
        shape[-2] = len(self.subset_indices[subset_index])
        return tuple(shape)

    @property
    def projection_geometry(self) -> Parallel2DGeometry | Parallel3DGeometry:
        """Fixed acquisition geometry supplied at construction."""
        return self._projection_geometry

    @property
    def volume_geometry(self) -> Volume2DGeometry | Volume3DGeometry:
        """Fixed working-grid geometry supplied at construction."""
        return self._volume_geometry

    @property
    def domain_shape(self) -> tuple[int, ...]:
        """Volume shape in (Y, X) or (Z, Y, X) order."""
        return self.volume_geometry.shape

    @property
    def range_shape(self) -> tuple[int, ...]:
        """Projection shape in (angles, detX) or (detY, angles, detX) order."""
        return self.projection_geometry.projection_shape

    @property
    def device_index(self) -> int:
        """CUDA device used by this operator."""
        return self._device_index

    @_on_device(cp)
    def forward(
        self,
        volume: np.ndarray | cp.ndarray,
        *,
        subset_index: int | None = None,
        out: np.ndarray | cp.ndarray | None = None,
    ) -> np.ndarray | cp.ndarray:
        """Project a volume into the acquisition geometry.

        Args:
            volume (numpy.ndarray or cupy.ndarray): Real data with domain_shape.
                Noncontiguous or non-float32 input is packed as float32.

        Keyword Args:
            subset_index (int or None): Default None selects all views, even
                when multiple subsets are configured. An integer selects one
                interleaved subset by its zero-based subset_indices index.
            out (numpy.ndarray or cupy.ndarray or None): Default None allocates
                a fresh float32 array with range_shape, or subset_shape(subset_index)
                when a subset is selected. Otherwise supply writable,
                C-contiguous float32 storage with that exact shape, using the same
                array library and device as the input. It must not overlap the
                input. Existing contents are overwritten; no accumulation occurs.

        Returns:
            numpy.ndarray or cupy.ndarray: Float32 projections using the input
            array library. When out is supplied, returns that array by identity.
            Input storage is preserved. Calls after close() raise RuntimeError.
        """
        if subset_index is not None:
            self.subset_shape(subset_index)
        options = {} if out is None else {"out": out}
        return self._backend.forward(volume, subset_index=subset_index, **options)

    @_on_device(cp)
    def backproject(
        self,
        projections: np.ndarray | cp.ndarray,
        *,
        subset_index: int | None = None,
        out: np.ndarray | cp.ndarray | None = None,
    ) -> np.ndarray | cp.ndarray:
        """Backproject measured projections using the selected backend conventions.

        Args:
            projections (numpy.ndarray or cupy.ndarray): Real data with
                range_shape, or subset_shape(subset_index) for a selected subset.
                Noncontiguous or non-float32 input is packed as float32. Supply
                only the selected views, in subset_indices acquisition order.

        Keyword Args:
            subset_index (int or None): Default None selects all views, even
                when multiple subsets are configured. An integer selects one
                interleaved subset by its zero-based subset_indices index.
            out (numpy.ndarray or cupy.ndarray or None): Default None allocates
                a fresh float32 array with domain_shape. Otherwise supply writable,
                C-contiguous float32 storage with domain_shape, using the same
                array library and device as the input. It must not overlap the
                input. Existing contents are overwritten; no accumulation occurs.

        Returns:
            numpy.ndarray or cupy.ndarray: Float32 volume using the input array
            library. When out is supplied, returns that array by identity. Input
            storage is preserved. Calls after close() raise RuntimeError.
        """
        if subset_index is not None:
            self.subset_shape(subset_index)
        options = {} if out is None else {"out": out}
        return self._backend.backproject(
            projections, subset_index=subset_index, **options
        )

    def close(self) -> None:
        """Release cached projectors and disable execution; repeated calls are safe."""
        self._backend.close()

    def __enter__(self) -> "ProjectionOperator":
        self._backend._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
