"""Geometry restrictions and detector phase convention for Fourier methods."""

import numpy as np

from tomobar.geometry import Parallel2DGeometry, Parallel3DGeometry


def require_fourier_geometry(
    projection_geometry: Parallel2DGeometry | Parallel3DGeometry, *, method: str
) -> None:
    """Allow angle scans and scalar CoR, but no flexible acquisition geometry."""
    if (
        isinstance(projection_geometry, Parallel3DGeometry)
        and not projection_geometry.is_standard_tomography
    ):
        raise ValueError(
            f"{method} requires standard Z-axis tomography and does not support "
            "vector geometries or laminography; use backend='astra' for flexible geometry"
        )
    if np.any(projection_geometry.detector_shifts != 0):
        raise ValueError(
            f"{method} does not support detector_shifts (including constant shifts); "
            "use centre_of_rotation for a scalar CoR correction, or backend='astra' "
            "for detector shifts"
        )
