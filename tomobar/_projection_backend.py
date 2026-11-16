"""Internal projector contract and construction shared by public entry points.

Importing this module does not import ASTRA or CuPy. Backend resources remain
lazy; geometry/type validation happens before any projection is executed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias

import numpy as np

from tomobar.geometry import (
    Parallel2DGeometry,
    Parallel3DGeometry,
    Volume2DGeometry,
    Volume3DGeometry,
)

if TYPE_CHECKING:
    from cupy import ndarray as GPUArray

BackendName: TypeAlias = Literal["astra", "usfft"]
ProjectionGeometry: TypeAlias = Parallel2DGeometry | Parallel3DGeometry
VolumeGeometry: TypeAlias = Volume2DGeometry | Volume3DGeometry
ProjectionArray: TypeAlias = "np.ndarray | GPUArray"


class ProjectionBackend(Protocol):
    """Fixed-geometry operations owned by one operator or reconstructor.

    Execution preserves the input's NumPy/CuPy library and packs real inputs as
    float32. Optional out storage must be independent, writable, contiguous
    float32 on the same device/library; it is overwritten and returned by
    identity. None selects all views, even when subsets are configured.
    Implementations may allocate temporary results when out is supplied.
    close() is idempotent; subsequent execution must fail. Concurrent calls or
    cleanup of one instance are unsupported. Numerical conventions are backend
    specific, including whether backprojection is an exact discrete transpose.
    """

    backend_name: BackendName

    @property
    def device_index(self) -> int:
        """CUDA device used for execution and input/output validation."""
        ...

    @property
    def subset_count(self) -> int:
        """Number of configured nonempty acquisition-order view partitions."""
        ...

    def forward(
        self,
        volume: ProjectionArray,
        *,
        subset_index: int | None = None,
        out: ProjectionArray | None = None,
    ) -> ProjectionArray:
        """Project a volume, optionally selecting one interleaved subset."""
        ...

    def backproject(
        self,
        projections: ProjectionArray,
        *,
        subset_index: int | None = None,
        out: ProjectionArray | None = None,
    ) -> ProjectionArray:
        """Backproject full data or the selected subset's data."""
        ...

    def _ensure_open(self) -> None:
        """Raise RuntimeError after resource ownership has ended."""
        ...

    def close(self) -> None:
        """Release owned resources without clearing shared CUDA caches."""
        ...


def _astra_backend_types() -> tuple[type[ProjectionBackend], type[ProjectionBackend]]:
    """Resolve ASTRA implementation types only when ASTRA is selected."""
    from tomobar.astra_wrappers.astra_tools2d import AstraTools2D
    from tomobar.astra_wrappers.astra_tools3d import AstraTools3D

    return AstraTools2D, AstraTools3D


def create_projection_backend(
    *,
    backend: BackendName,
    projection_geometry: ProjectionGeometry,
    volume_geometry: VolumeGeometry,
    device_index: int = 0,
    subset_count: int = 1,
) -> ProjectionBackend:
    """Validate the selection and construct a backend without execution.

    Matching 2D/3D geometry types are required. ASTRA accepts general supported
    acquisition geometry; USFFT applies its additional geometry restrictions.
    This function owns the translation from neutral subset_count to ASTRA's
    internal legacy constructor argument. The caller owns close() on success.
    Invalid selections/types raise ValueError/TypeError before resource creation.
    """
    if backend not in ("astra", "usfft"):
        raise ValueError("backend must be astra or usfft")
    if isinstance(projection_geometry, Parallel3DGeometry):
        volume_type = Volume3DGeometry
    elif isinstance(projection_geometry, Parallel2DGeometry):
        volume_type = Volume2DGeometry
    else:
        raise TypeError(
            "projection_geometry must be a Parallel2DGeometry or Parallel3DGeometry"
        )
    if not isinstance(volume_geometry, volume_type):
        raise TypeError(f"volume_geometry must be a {volume_type.__name__}")
    if backend == "usfft":
        from tomobar._usfft_adapter import USFFTGeometryBackend

        return USFFTGeometryBackend(
            projection_geometry=projection_geometry,
            volume_geometry=volume_geometry,
            device_index=device_index,
            subset_count=subset_count,
        )
    tools2d, tools3d = _astra_backend_types()
    wrapper = (
        tools3d if isinstance(projection_geometry, Parallel3DGeometry) else tools2d
    )
    return wrapper(
        projection_geometry=projection_geometry,
        volume_geometry=volume_geometry,
        device_index=device_index,
        ordsub_number=subset_count,
    )
