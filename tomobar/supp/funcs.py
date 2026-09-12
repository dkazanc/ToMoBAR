"""Supporting functions"""

import numpy as np
from typing import Union, List, Tuple

cupy_enabled = False
try:
    import cupy as xp

    try:
        xp.cuda.Device(0).compute_capability
        cupy_enabled = True
    except xp.cuda.runtime.CUDARuntimeError:
        import numpy as xp

        print("CuPy is installed but the GPU device is inaccessible")
except ImportError:
    import numpy as xp


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


def _normalise_center_rot_offset(offset, n_angles, *, is_3d):
    """Validate detector offsets, retaining scalar and per-view representations.

    None means zero. A scalar or (n_angles,) array specifies horizontal
    displacement; 3D additionally accepts (n_angles, 2) horizontal/vertical
    pairs. Values are finite, real numbers in geometry length units (pixels
    for ToMoBAR's unit detector spacing). A length-two vector always means
    two horizontal offsets, never a constant horizontal/vertical pair.
    """
    if offset is None:
        return 0.0
    values = np.asarray(offset)
    if values.dtype.kind not in "iuf":
        raise TypeError("CenterRotOffset must contain real numeric values")
    allowed_shapes = [(), (n_angles,)]
    if is_3d:
        allowed_shapes.append((n_angles, 2))
    if values.shape not in allowed_shapes:
        raise ValueError(
            f"CenterRotOffset must have shape {allowed_shapes}, got {values.shape}"
        )
    values = np.array(values, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(values)):
        raise ValueError("CenterRotOffset must contain only finite values")
    return float(values) if values.ndim == 0 else values


# define a 2D vector geometry
def _vec_geom_init2D(
    angles_rad: np.ndarray, CenterRotOffset: Union[float, List]
) -> np.ndarray:
    CenterRotOffset = _normalise_center_rot_offset(
        CenterRotOffset, angles_rad.size, is_3d=False
    )
    DetectorSpacingX = 1.0
    s0 = [0.0, -1.0]  # source
    u0 = [DetectorSpacingX, 0.0]  # detector coordinates
    vectors = np.zeros([angles_rad.size, 6])
    for i in range(0, angles_rad.size):
        if np.ndim(CenterRotOffset) == 0:
            d0 = [CenterRotOffset, 0.0]  # detector
        else:
            d0 = [CenterRotOffset[i], 0.0]  # detector
        theta = angles_rad[i]
        vec_temp = np.dot(__rotation_matrix2D(theta), s0)
        vectors[i, 0:2] = vec_temp[:]  # ray position
        vec_temp = np.dot(__rotation_matrix2D(theta), d0)
        vectors[i, 2:4] = vec_temp[:]  # center of detector position
        vec_temp = np.dot(__rotation_matrix2D(theta), u0)
        vectors[i, 4:6] = vec_temp[:]  # detector pixel (0,0) to (0,1).
    return vectors


# define 3D vector geometry
def _vec_geom_init3D(angles_rad, DetectorSpacingX, DetectorSpacingY, CenterRotOffset):
    """Build 3D vectors from scalar, (n,) horizontal or (n, 2) H/V offsets.

    Positive horizontal displacement follows the detector column direction;
    positive vertical displacement follows detector rows (world +Z for CT).
    Offsets are geometry lengths, independent of the detector pixel spacing.
    """
    CenterRotOffset = _normalise_center_rot_offset(
        CenterRotOffset, angles_rad.size, is_3d=True
    )
    s0 = [0.0, -1.0, 0.0]  # source
    u0 = [DetectorSpacingX, 0.0, 0.0]  # detector coordinates
    v0 = [0.0, 0.0, DetectorSpacingY]  # detector coordinates

    vectors = np.zeros([angles_rad.size, 12])
    for i in range(0, angles_rad.size):
        if np.ndim(CenterRotOffset) == 0:
            d0 = [CenterRotOffset, 0.0, 0.0]  # detector
        elif np.ndim(CenterRotOffset) == 1:
            d0 = [CenterRotOffset[i], 0.0, 0.0]  # horizontal offset per view
        else:
            d0 = [CenterRotOffset[i, 0], 0.0, CenterRotOffset[i, 1]]  # detector
        theta = angles_rad[i]
        vec_temp = np.dot(__rotation_matrix3D(theta), s0)
        vectors[i, 0:3] = vec_temp[:]  # ray position
        vec_temp = np.dot(__rotation_matrix3D(theta), d0)
        vectors[i, 3:6] = vec_temp[:]  # center of detector position
        vec_temp = np.dot(__rotation_matrix3D(theta), u0)
        vectors[i, 6:9] = vec_temp[:]  # detector pixel (0,0) to (0,1).
        vec_temp = np.dot(__rotation_matrix3D(theta), v0)
        vectors[i, 9:12] = vec_temp[:]  # Vector from detector pixel (0,0) to (1,0)
    return vectors


# define 2D rotation matrix
def __rotation_matrix2D(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


# define 3D rotation matrix
def __rotation_matrix3D(theta):
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def __get_swap_tuple(data_axis_labels, labels_order):
    swap_tuple = None
    for in_l1, str_1 in enumerate(labels_order):
        for in_l2, str_2 in enumerate(data_axis_labels):
            if str_1 == str_2:
                # get the indices only IF the order is different
                if in_l1 != in_l2:
                    swap_tuple = (in_l1, in_l2)
                    return swap_tuple
    return swap_tuple


def _axis_permutation(source_axes, target_axes):
    """Return source-to-target indices for a complete 2D/3D projection layout."""
    for name, axes in (("source_axes", source_axes), ("target_axes", target_axes)):
        if not isinstance(axes, (list, tuple)) or not all(
            isinstance(axis, str) for axis in axes
        ):
            raise ValueError(f"{name} must be a list or tuple of axis labels")
        if len(axes) not in (2, 3) or len(set(axes)) != len(axes):
            raise ValueError(f"{name} must contain two or three unique axis labels")
        expected = {"angles", "detX"} if len(axes) == 2 else {"detY", "angles", "detX"}
        if set(axes) != expected:
            raise ValueError(f"{name} must contain exactly {sorted(expected)}")
    if len(source_axes) != len(target_axes):
        raise ValueError("Source and target axes must have the same length")
    return tuple(source_axes.index(axis) for axis in target_axes)


def _transpose_data_axes(data, *, source_axes, target_axes):
    """Reorder NumPy/CuPy arrays or estimator shape tuples without copying."""
    permutation = _axis_permutation(source_axes, target_axes)
    ndim = len(data) if isinstance(data, tuple) else data.ndim
    if ndim != len(permutation):
        raise ValueError(
            f"Axis labels describe {len(permutation)} dimensions, got {ndim}"
        )
    if permutation == tuple(range(ndim)):
        return data
    if isinstance(data, tuple):
        return tuple(data[index] for index in permutation)
    return data.transpose(permutation)


def _resolve_data_axes(explicit_axes, legacy_axes, default_axes):
    """Resolve explicit input/output axes and the legacy keyword consistently."""
    if explicit_axes is not None:
        _axis_permutation(explicit_axes, default_axes)
    if legacy_axes is not None:
        _axis_permutation(legacy_axes, default_axes)
    if explicit_axes is not None and legacy_axes is not None:
        if tuple(explicit_axes) != tuple(legacy_axes):
            raise ValueError("Conflicting axis orders in explicit and legacy keywords")
    return (
        explicit_axes
        if explicit_axes is not None
        else (legacy_axes if legacy_axes is not None else default_axes)
    )


def _swap_data_axes_to_accepted(data_axes_labels, required_labels_order):
    """A module to ensure that the input tomographic data is prepeared for reconstruction
    in the axes order required.

    Args:
        data_axes_labels (list):  a list of data labels, e.g. given as ['angles', 'detX', 'detY']
        required_labels_order (list): the required (fixed) order of axis labels for data, e.g. ["detY", "angles", "detX"].

    Returns:
        list: A list of two tuples for input data swaping axis. If both are None, then no swapping needed.
    """

    _axis_permutation(data_axes_labels, required_labels_order)
    swap_tuple2 = None
    # check if the labels names are the accepted ones
    for str_1 in data_axes_labels:
        if str_1 not in required_labels_order:
            raise ValueError(
                f'Axis title "{str_1}" is not valid, please use one of these: "angles", "detX", or "detY"'
            )
    data_axes_labels_copy = list(data_axes_labels)

    # check the order and produce a swapping tuple if needed
    swap_tuple1 = __get_swap_tuple(data_axes_labels_copy, required_labels_order)

    if swap_tuple1 is not None:
        # swap elements in the list and check the list again
        data_axes_labels_copy[swap_tuple1[0]], data_axes_labels_copy[swap_tuple1[1]] = (
            data_axes_labels_copy[swap_tuple1[1]],
            data_axes_labels_copy[swap_tuple1[0]],
        )
        swap_tuple2 = __get_swap_tuple(data_axes_labels_copy, required_labels_order)

    if swap_tuple2 is not None:
        # swap elements in the list
        data_axes_labels_copy[swap_tuple2[0]], data_axes_labels_copy[swap_tuple2[1]] = (
            data_axes_labels_copy[swap_tuple2[1]],
            data_axes_labels_copy[swap_tuple2[0]],
        )

    return [swap_tuple1, swap_tuple2]


def swap_tuple_elements(tup: Tuple[int, int, int], idx1: int, idx2: int):
    items = list(tup)
    items[idx1], items[idx2] = items[idx2], items[idx1]
    return tuple(items)


def _data_swap(
    data: xp.ndarray | Tuple[int, int, int], data_swap_list: list
) -> xp.ndarray:
    """Swap data labels based on the provided list of tuples

    Args:
        data (xp.ndarray): Numpy or CuPu 2D or 3D array
        data_swap_list (list): List of tuples to swap to

    Returns:
        xp.ndarray: swapped array to the desired format
    """
    for swap_tuple in data_swap_list:
        if swap_tuple is not None:
            if isinstance(data, tuple):
                data = swap_tuple_elements(data, swap_tuple[0], swap_tuple[1])
            else:
                if cupy_enabled:
                    xpp = xp.get_array_module(data)
                    data = xpp.swapaxes(data, swap_tuple[0], swap_tuple[1])
                else:
                    data = np.swapaxes(data, swap_tuple[0], swap_tuple[1])

    return data


def _parse_device_argument(device_int_or_string) -> Tuple:
    """Convert a cpu/gpu string or integer gpu number into a tuple."""
    if isinstance(device_int_or_string, int):
        return "gpu", device_int_or_string
    elif device_int_or_string == "gpu":
        return "gpu", 0
    elif device_int_or_string == "cpu":
        return "cpu", -1
    else:
        raise ValueError(
            'Unknown device {0}. Expecting either "cpu" or "gpu" strings OR the gpu device integer'.format(
                device_int_or_string
            )
        )


def _data_dims_swapper(
    data: xp.ndarray, data_axes_labels_order: list, required_labels_order: list
) -> xp.ndarray:
    """Swaps data axes as it required

    Args:
        data (xp.ndarray): 2D or 3D array.
        data_axes_labels_order (list): The input data axes.
        required_labels_order (list): The required data axes.

    Returns:
        xp.ndarray: An array with swapped (or not) axes.
    """
    return _transpose_data_axes(
        data, source_axes=data_axes_labels_order, target_axes=required_labels_order
    )
