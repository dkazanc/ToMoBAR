"""Projection axis ordering for NumPy/CuPy arrays and estimator shapes."""

from typing import Any, Tuple


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


def _data_swap(data: Any | Tuple[int, int, int], data_swap_list: list) -> Any:
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
                data = data.swapaxes(swap_tuple[0], swap_tuple[1])

    return data
