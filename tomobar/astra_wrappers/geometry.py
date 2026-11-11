"""Convert backend-independent geometry to ASTRA dictionaries without resources.

These helpers need neither ASTRA nor CUDA. Projector creation and ownership
belong to the wrapper that binds the resulting dictionaries.
"""

import numpy as np

from tomobar.geometry import (
    Parallel2DGeometry,
    Parallel3DGeometry,
    Volume2DGeometry,
    VolumeGeometry,
)


def parallel2d_vectors(geometry: Parallel2DGeometry) -> np.ndarray:
    """Build acquisition-order 2D vectors with unit detector spacing."""
    sine, cosine = np.sin(geometry.angles), np.cos(geometry.angles)
    horizontal = geometry.detector_shifts + geometry.centre_of_rotation
    return np.column_stack(
        (sine, -cosine, horizontal * cosine, horizontal * sine, cosine, sine)
    )


def parallel3d_vectors(geometry: Parallel3DGeometry) -> np.ndarray:
    """Build all views in acquisition order with unit detector pixel spacing."""
    sine = np.sin(geometry.angles)
    cosine = np.cos(geometry.angles)
    horizontal, vertical = geometry.detector_shifts.T
    horizontal = horizontal + geometry.centre_of_rotation
    vectors = np.zeros((geometry.angles.size, 12), dtype=np.float64)
    vectors[:, 0] = sine
    vectors[:, 1] = -cosine
    vectors[:, 3] = horizontal * cosine
    vectors[:, 4] = horizontal * sine
    vectors[:, 5] = vertical
    vectors[:, 6] = cosine
    vectors[:, 7] = sine
    vectors[:, 11] = 1.0
    return vectors


def projection_geometry(
    geometry: Parallel2DGeometry | Parallel3DGeometry,
    vectors: np.ndarray,
    view_indices: np.ndarray | None = None,
) -> dict:
    """Convert a scan, selecting subset rows from precomputed full-view vectors."""
    if isinstance(geometry, Parallel2DGeometry):
        columns = geometry.detector_shape[0]
        return {
            "type": "parallel_vec",
            "DetectorCount": columns,
            "Vectors": vectors if view_indices is None else vectors[view_indices],
        }
    rows, columns = geometry.detector_shape
    return {
        "type": "parallel3d_vec",
        "DetectorRowCount": rows,
        "DetectorColCount": columns,
        "Vectors": vectors if view_indices is None else vectors[view_indices],
    }


def volume_geometry(geometry: Volume2DGeometry | VolumeGeometry) -> dict:
    """Convert (y, x) or (z, y, x) voxels to a centred ASTRA volume."""
    y, x = geometry.shape[-2:]
    converted = {
        "GridRowCount": y,
        "GridColCount": x,
        "option": {
            "WindowMinX": -x / 2,
            "WindowMaxX": x / 2,
            "WindowMinY": -y / 2,
            "WindowMaxY": y / 2,
        },
    }
    if isinstance(geometry, VolumeGeometry):
        z = geometry.shape[0]
        converted["GridSliceCount"] = z
        converted["option"].update(WindowMinZ=-z / 2, WindowMaxZ=z / 2)
    return converted
