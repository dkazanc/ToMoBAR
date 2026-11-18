"""Projection axis ordering for NumPy/CuPy arrays and estimator shapes."""


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


def _resolve_data_axes(explicit_axes, default_axes):
    """Validate explicit input/output axes or use the default layout."""
    if explicit_axes is not None:
        _axis_permutation(explicit_axes, default_axes)
    return explicit_axes if explicit_axes is not None else default_axes
