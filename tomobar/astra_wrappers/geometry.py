"""Convert backend-independent geometry to ASTRA dictionaries without resources.

These helpers need neither ASTRA nor CUDA. Projector creation and ownership
belong to the wrapper that binds the resulting dictionaries.
"""

import numpy as np

from tomobar.geometry import Parallel3DGeometry, VolumeGeometry


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
    geometry: Parallel3DGeometry,
    vectors: np.ndarray,
    view_indices: np.ndarray | None = None,
) -> dict:
    """Convert a scan, selecting subset rows from precomputed full-view vectors."""
    rows, columns = geometry.detector_shape
    return {
        "type": "parallel3d_vec",
        "DetectorRowCount": rows,
        "DetectorColCount": columns,
        "Vectors": vectors if view_indices is None else vectors[view_indices],
    }


def volume_geometry(geometry: VolumeGeometry) -> dict:
    """Convert (z, y, x) voxels to ASTRA's centred, unit-spaced volume."""
    z, y, x = geometry.shape
    return {
        "GridRowCount": y,
        "GridColCount": x,
        "GridSliceCount": z,
        "option": {
            "WindowMinX": -x / 2,
            "WindowMaxX": x / 2,
            "WindowMinY": -y / 2,
            "WindowMaxY": y / 2,
            "WindowMinZ": -z / 2,
            "WindowMaxZ": z / 2,
        },
    }
