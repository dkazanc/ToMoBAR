"""Backend-independent geometry for 3D tomography."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_shape(shape, ndim: int, name: str) -> tuple[int, ...]:
    dimensions = tuple(shape)
    if len(dimensions) != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if any(
        isinstance(size, (bool, np.bool_))
        or not isinstance(size, (int, np.integer))
        for size in dimensions
    ):
        raise TypeError(f"{name} dimensions must be integers")
    if any(size <= 0 for size in dimensions):
        raise ValueError(f"{name} dimensions must be positive")
    return tuple(int(size) for size in dimensions)


def _freeze_array(values: np.ndarray, name: str) -> NDArray[np.float64]:
    if values.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numbers")
    with np.errstate(over="ignore"):
        values = np.asarray(values, dtype=np.float64)
    # Immutable storage keeps the geometry independent of caller-owned arrays.
    frozen = np.frombuffer(values.tobytes(), dtype=np.float64).reshape(values.shape)
    if not np.isfinite(frozen).all():
        raise ValueError(f"{name} must contain only finite values")
    return frozen


@dataclass(frozen=True, eq=False, init=False)
class Parallel3DGeometry:
    """Parallel-beam acquisition rotating around the volume's Z axis.

    Angles are in radians and stored in acquisition order. The detector shape
    is (rows, columns). Detector shifts are (horizontal, vertical) pairs in
    pixels, one pair per view. Positive shifts move the detector centre along
    the increasing column and row directions, respectively.
    """

    angles: NDArray[np.float64]
    detector_shape: tuple[int, int]
    detector_shifts: NDArray[np.float64]

    def __init__(
        self,
        *,
        angles: ArrayLike,
        detector_shape: tuple[int, int],
        detector_shifts: ArrayLike | None = None,
    ) -> None:
        angle_values = np.asarray(angles)
        if angle_values.ndim != 1 or angle_values.size == 0:
            raise ValueError("angles must be a nonempty one-dimensional array")
        rows, columns = _validate_shape(detector_shape, 2, "detector_shape")

        shift_shape = (angle_values.size, 2)
        shifts = (
            np.zeros(shift_shape)
            if detector_shifts is None
            else np.asarray(detector_shifts)
        )
        if shifts.shape != shift_shape:
            raise ValueError(f"detector_shifts must have shape {shift_shape}")

        object.__setattr__(self, "angles", _freeze_array(angle_values, "angles"))
        object.__setattr__(self, "detector_shape", (rows, columns))
        object.__setattr__(
            self, "detector_shifts", _freeze_array(shifts, "detector_shifts")
        )

    @classmethod
    def tomography(
        cls,
        *,
        angles: ArrayLike,
        detector_shape: tuple[int, int],
        detector_shifts: ArrayLike | None = None,
    ) -> "Parallel3DGeometry":
        """Build a scan geometry, using zero shifts when they are omitted."""
        return cls(
            angles=angles,
            detector_shape=detector_shape,
            detector_shifts=detector_shifts,
        )


@dataclass(frozen=True)
class VolumeGeometry:
    """A centred, unit-spaced 3D voxel grid with shape (z, y, x)."""

    shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", _validate_shape(self.shape, 3, "shape"))
