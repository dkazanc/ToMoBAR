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
    """Parallel acquisition from Z-axis scan angles or general 3D vectors.

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

    angles: NDArray[np.float64] | None
    vectors: NDArray[np.float64] | None
    detector_shape: tuple[int, int]
    detector_shifts: NDArray[np.float64]
    centre_of_rotation: float

    def __init__(
        self,
        *,
        angles: ArrayLike | None = None,
        vectors: ArrayLike | None = None,
        detector_shape: tuple[int, int],
        detector_shifts: ArrayLike | None = None,
        centre_of_rotation: float = 0.0,
    ) -> None:
        if (angles is None) == (vectors is None):
            raise ValueError("provide exactly one of angles or vectors")
        angle_values = None
        vector_values = None
        if vectors is not None:
            vector_values = _freeze_array(np.asarray(vectors), "vectors")
            if (
                vector_values.ndim != 2
                or vector_values.shape[1] != 12
                or not len(vector_values)
            ):
                raise ValueError(
                    "vectors must have shape (n_views, 12), with n_views > 0"
                )
            ray, u, v = (
                vector_values[:, :3],
                vector_values[:, 6:9],
                vector_values[:, 9:12],
            )
            # Test directions after normalisation to avoid dependence on units.
            norms = [np.linalg.norm(q, axis=1) for q in (ray, u, v)]
            if any(np.any(n == 0) or not np.isfinite(n).all() for n in norms):
                raise ValueError(
                    "ray and detector basis vectors must be nonzero with finite norms"
                )
            r, un, vn = (q / n[:, None] for q, n in zip((ray, u, v), norms))
            if np.any(np.abs(np.sum(r * np.cross(un, vn), axis=1)) < 1e-10):
                raise ValueError("rays must intersect a nondegenerate detector plane")
            count = len(vector_values)
        else:
            angle_values = np.asarray(angles)
            if angle_values.ndim != 1 or angle_values.size == 0:
                raise ValueError("angles must be a nonempty one-dimensional array")
            angle_values = _freeze_array(angle_values, "angles")
            count = angle_values.size
        rows, columns = _validate_shape(detector_shape, 2, "detector_shape")
        shift_shape = (count, 2)
        shifts = (
            np.zeros(shift_shape)
            if detector_shifts is None
            else np.asarray(detector_shifts)
        )
        if shifts.shape != shift_shape:
            raise ValueError(f"detector_shifts must have shape {shift_shape}")

        object.__setattr__(self, "angles", angle_values)
        object.__setattr__(self, "vectors", vector_values)
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

    @classmethod
    def from_vectors(
        cls,
        *,
        vectors: ArrayLike,
        detector_shape: tuple[int, int],
        detector_shifts: ArrayLike | None = None,
        centre_of_rotation: float = 0.0,
    ) -> "Parallel3DGeometry":
        """Construct from (ray, detector centre, column step, row step) rows.

        Coordinates and step lengths use the volume's unit voxel coordinates.
        Shifts and CoR are additional detector-pixel offsets along the supplied
        basis. Angles are absent: acquisition order is the vector row order.
        """
        return cls(
            vectors=vectors,
            detector_shape=detector_shape,
            detector_shifts=detector_shifts,
            centre_of_rotation=centre_of_rotation,
        )

    @classmethod
    def laminography(
        cls,
        *,
        angles: ArrayLike,
        detector_shape: tuple[int, int],
        tilt_angle: float,
        rotation_axis: ArrayLike = (0.0, 0.0, 1.0),
        detector_shifts: ArrayLike | None = None,
        centre_of_rotation: float = 0.0,
    ) -> "Parallel3DGeometry":
        """Parallel laminography in a sample-fixed frame, angles in radians.

        Tilt angle is the angle between ray and rotation axis, in (0, pi/2].
        pi/2 gives conventional tomography for the default +Z axis. The axis
        passes through the volume origin. At zero rotation, detector columns
        follow +X projected onto the axis-normal plane (+Y if degenerate).
        Positive scan angles rotate the acquisition frame right-handed about
        the axis. Detector pixels have unit spacing and an orthogonal basis.
        """
        value = np.asarray(tilt_angle)
        if value.ndim != 0 or value.dtype.kind not in "iuf":
            raise TypeError("tilt_angle must be a real scalar in radians")
        tilt_angle = float(value)
        if not np.isfinite(tilt_angle) or not 0 < tilt_angle <= np.pi / 2:
            raise ValueError("tilt_angle must be in (0, pi/2]")
        axis = _freeze_array(np.asarray(rotation_axis), "rotation_axis")
        if axis.shape != (3,) or not 0 < np.linalg.norm(axis) < np.inf:
            raise ValueError("rotation_axis must be a nonzero three-vector")
        axis = axis / np.linalg.norm(axis)
        base = np.array([1.0, 0.0, 0.0])
        if abs(axis[0]) > 0.99:
            base = np.array([0.0, 1.0, 0.0])
        e1 = base - np.dot(base, axis) * axis
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(axis, e1)
        scan = cls.tomography(
            angles=angles,
            detector_shape=detector_shape,
            detector_shifts=detector_shifts,
            centre_of_rotation=centre_of_rotation,
        )
        if tilt_angle == np.pi / 2 and np.array_equal(axis, [0.0, 0.0, 1.0]):
            return scan
        sine, cosine = np.sin(scan.angles)[:, None], np.cos(scan.angles)[:, None]
        u = cosine * e1 + sine * e2
        ray = np.sin(tilt_angle) * (sine * e1 - cosine * e2) + np.cos(tilt_angle) * axis
        v = np.cross(ray, u)
        vectors = np.column_stack((ray, np.zeros_like(ray), u, v))
        return cls.from_vectors(
            vectors=vectors,
            detector_shape=detector_shape,
            detector_shifts=scan.detector_shifts,
            centre_of_rotation=centre_of_rotation,
        )

    @property
    def view_count(self) -> int:
        return len(self.vectors) if self.vectors is not None else self.angles.size

    @property
    def is_standard_tomography(self) -> bool:
        """Whether the existing slice-based analytical methods are supported."""
        return self.vectors is None

    def transformed(
        self, *, rotation: ArrayLike, translation: ArrayLike = (0.0, 0.0, 0.0)
    ) -> "Parallel3DGeometry":
        """Rigidly transform the acquisition frame about the volume origin.

        ``rotation`` is a proper 3x3 rotation acting on column vectors in XYZ
        coordinates. Translation moves detector centres in voxel units; ray
        directions and pixel basis vectors only rotate. Existing pixel shifts
        are incorporated before transformation, so they are not applied twice.
        The result is explicit-vector geometry with no scan-angle metadata.
        """
        matrix = _freeze_array(np.asarray(rotation), "rotation")
        offset = _freeze_array(np.asarray(translation), "translation")
        if (
            matrix.shape != (3, 3)
            or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-10, rtol=0)
            or not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-10, rtol=0)
        ):
            raise ValueError("rotation must be a proper orthogonal 3x3 matrix")
        if offset.shape != (3,):
            raise ValueError("translation must have shape (3,)")
        vectors = self.to_vectors().reshape(self.view_count, 4, 3) @ matrix.T
        vectors[:, 1, :] += offset
        return type(self).from_vectors(
            vectors=vectors.reshape(self.view_count, 12),
            detector_shape=self.detector_shape,
        )

    def to_vectors(self) -> NDArray[np.float64]:
        """Return fresh vectors with detector shifts and CoR applied once."""
        if self.vectors is None:
            sine, cosine = np.sin(self.angles), np.cos(self.angles)
            vectors = np.zeros((self.view_count, 12), dtype=np.float64)
            vectors[:, 0], vectors[:, 1] = sine, -cosine
            vectors[:, 6], vectors[:, 7], vectors[:, 11] = cosine, sine, 1.0
        else:
            vectors = self.vectors.copy()
        horizontal, vertical = self.detector_shifts.T
        vectors[:, 3:6] += (horizontal + self.centre_of_rotation)[:, None] * vectors[
            :, 6:9
        ] + vertical[:, None] * vectors[:, 9:12]
        return vectors

    @property
    def projection_shape(self) -> tuple[int, int, int]:
        """Native projection array shape (detector rows, views, columns)."""
        rows, columns = self.detector_shape
        return rows, self.view_count, columns


@dataclass(frozen=True)
class Volume3DGeometry:
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
