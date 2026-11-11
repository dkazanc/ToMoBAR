"""Backend-independent geometry for 3D tomography."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_ordsub_number(value, n_angles):
    """Normalize None to one; subsets must be nonempty integer partitions."""
    if value is None:
        value = 1
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("The number of ordered subsets must be an integer")
    if not 1 <= value <= n_angles:
        raise ValueError(
            f"The number of ordered subsets must be between 1 and {n_angles}"
        )
    return int(value)


def interleaved_view_indices(
    view_count: int, subset_count: int | None = None
) -> tuple[NDArray[np.intp], ...]:
    """Partition acquisition-order views into nonempty, read-only subsets."""
    (view_count,) = _validate_shape((view_count,), 1, "view_count")
    count = _validate_ordsub_number(subset_count, view_count)
    return tuple(
        np.frombuffer(
            np.arange(index, view_count, count, dtype=np.intp).tobytes(), dtype=np.intp
        )
        for index in range(count)
    )


def _validate_shape(shape, ndim: int, name: str) -> tuple[int, ...]:
    dimensions = tuple(shape)
    if len(dimensions) != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if any(
        isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer))
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


def _validate_centre_of_rotation(value: float) -> float:
    """Require one finite horizontal CoR correction in detector pixels."""
    if value is None:
        raise TypeError(
            "centre_of_rotation must be a real scalar, not None; "
            "use 0.0 for a centred rotation axis"
        )
    scalar = np.asarray(value)
    if scalar.dtype.kind not in "iuf":
        raise TypeError("centre_of_rotation must be a real scalar")
    if scalar.ndim != 0:
        raise ValueError(
            "centre_of_rotation must be a scalar; use detector_shifts for per-view displacements"
        )
    correction = float(scalar)
    if not np.isfinite(correction):
        raise ValueError("centre_of_rotation must be finite")
    return correction


@dataclass(frozen=True, eq=False, init=False)
class Parallel3DGeometry:
    """Parallel-beam acquisition rotating around the volume's Z axis.

    Angles are in radians and stored in acquisition order. The detector shape
    is (rows, columns). Detector shifts are (horizontal, vertical) pairs in
    pixels, one pair per view. Positive shifts move the detector centre along
    the increasing column and row directions, respectively.

    centre_of_rotation is a single horizontal CoR correction in pixels,
    independent of detector motion. Positive corrections move the effective
    detector centre along increasing columns. The effective horizontal
    displacement is detector_shifts[:, 0] + centre_of_rotation; vertical
    displacement is detector_shifts[:, 1]. Zero is a centred rotation axis.
    """

    angles: NDArray[np.float64]
    detector_shape: tuple[int, int]
    detector_shifts: NDArray[np.float64]
    centre_of_rotation: float

    def __init__(
        self,
        *,
        angles: ArrayLike,
        detector_shape: tuple[int, int],
        detector_shifts: ArrayLike | None = None,
        centre_of_rotation: float = 0.0,
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
        object.__setattr__(
            self, "centre_of_rotation", _validate_centre_of_rotation(centre_of_rotation)
        )

    @classmethod
    def tomography(
        cls,
        *,
        angles: ArrayLike,
        detector_shape: tuple[int, int],
        detector_shifts: ArrayLike | None = None,
        centre_of_rotation: float = 0.0,
    ) -> "Parallel3DGeometry":
        """Build a scan geometry, using zero shifts when they are omitted."""
        return cls(
            angles=angles,
            detector_shape=detector_shape,
            detector_shifts=detector_shifts,
            centre_of_rotation=centre_of_rotation,
        )

    @property
    def projection_shape(self) -> tuple[int, int, int]:
        """Native projection array shape (detector rows, views, columns)."""
        rows, columns = self.detector_shape
        return rows, self.angles.size, columns


@dataclass(frozen=True)
class VolumeGeometry:
    """A centred, unit-spaced 3D voxel grid with shape (z, y, x)."""

    shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", _validate_shape(self.shape, 3, "shape"))


@dataclass(frozen=True, eq=False, init=False)
class Parallel2DGeometry:
    """2D parallel scan with (columns,) detector shape and per-view shifts.

    Horizontal detector shifts and the scalar CoR correction are in pixels,
    positive along detector columns. Angles are in radians.
    """

    angles: NDArray[np.float64]
    detector_shape: tuple[int]
    detector_shifts: NDArray[np.float64]
    centre_of_rotation: float

    def __init__(
        self,
        *,
        angles: ArrayLike,
        detector_shape: tuple[int],
        detector_shifts: ArrayLike | None = None,
        centre_of_rotation: float = 0.0,
    ):
        angles = np.asarray(angles)
        if angles.ndim != 1 or angles.size == 0:
            raise ValueError("angles must be a nonempty one-dimensional array")
        shifts = (
            np.zeros(angles.size)
            if detector_shifts is None
            else np.asarray(detector_shifts)
        )
        if shifts.shape != angles.shape:
            raise ValueError("detector_shifts must have shape (n_views,)")
        object.__setattr__(self, "angles", _freeze_array(angles, "angles"))
        object.__setattr__(
            self, "detector_shape", _validate_shape(detector_shape, 1, "detector_shape")
        )
        object.__setattr__(
            self, "detector_shifts", _freeze_array(shifts, "detector_shifts")
        )
        object.__setattr__(
            self, "centre_of_rotation", _validate_centre_of_rotation(centre_of_rotation)
        )

    @property
    def projection_shape(self) -> tuple[int, int]:
        """Native projection array shape (views, detector columns)."""
        return self.angles.size, self.detector_shape[0]


@dataclass(frozen=True)
class Volume2DGeometry:
    """Centred, unit-spaced 2D grid with shape (y, x)."""

    shape: tuple[int, int]

    def __post_init__(self):
        object.__setattr__(self, "shape", _validate_shape(self.shape, 2, "shape"))
