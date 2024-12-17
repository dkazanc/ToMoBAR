"""Supporting functions

"""

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


# define a 2D vector geometry
def _vec_geom_init2D(
    angles_rad: np.ndarray, CenterRotOffset: Union[float, List]
) -> np.ndarray:
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
    s0 = [0.0, -1.0, 0.0]  # source
    u0 = [DetectorSpacingX, 0.0, 0.0]  # detector coordinates
    v0 = [0.0, 0.0, DetectorSpacingY]  # detector coordinates

    vectors = np.zeros([angles_rad.size, 12])
    for i in range(0, angles_rad.size):
        if np.ndim(CenterRotOffset) == 0:
            d0 = [CenterRotOffset, 0.0, 0.0]  # detector
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


def _swap_data_axes_to_accepted(data_axes_labels, required_labels_order):
    """A module to ensure that the input tomographic data is prepeared for reconstruction
    in the axes order required.

    Args:
        data_axes_labels (list):  a list of data labels, e.g. given as ['angles', 'detX', 'detY']
        required_labels_order (list): the required (fixed) order of axis labels for data, e.g. ["detY", "angles", "detX"].

    Returns:
        list: A list of two tuples for input data swaping axis. If both are None, then no swapping needed.
    """

    if len(data_axes_labels) != len(required_labels_order):
        raise ValueError(
            "Warning: The mismatch between provided labels and data dimensions."
        )
    swap_tuple2 = None
    # check if the labels names are the accepted ones
    for str_1 in data_axes_labels:
        if str_1 not in required_labels_order:
            raise ValueError(
                f'Axis title "{str_1}" is not valid, please use one of these: "angles", "detX", or "detY"'
            )
    data_axes_labels_copy = data_axes_labels.copy()

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


def _data_swap(data: xp.ndarray, data_swap_list: list) -> xp.ndarray:
    """Swap data labels based on the provided list of tuples

    Args:
        data (xp.ndarray): Numpy or CuPu 2D or 3D array
        data_swap_list (list): List of tuples to swap to

    Returns:
        xp.ndarray: swapped array to the desired format
    """
    for swap_tuple in data_swap_list:
        if swap_tuple is not None:
            if cupy_enabled:
                xpp = xp.get_array_module(data)
                return xpp.swapaxes(data, swap_tuple[0], swap_tuple[1])
            else:
                return np.swapaxes(data, swap_tuple[0], swap_tuple[1])
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
    data_swap_list = _swap_data_axes_to_accepted(
        data_axes_labels_order, required_labels_order
    )
    return _data_swap(data, data_swap_list)
