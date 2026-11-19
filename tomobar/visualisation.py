"""CPU-only geometry plots; Matplotlib is imported only when plotting."""

from itertools import product
from typing import TYPE_CHECKING, Sequence

import numpy as np

from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry

from tomobar._visualisation_laboratory import _plot_laboratory

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["plot_geometry"]


def plot_geometry(
    projection_geometry: Parallel3DGeometry,
    volume_geometry: Volume3DGeometry,
    *,
    view_indices: Sequence[int] | None = None,
    ax: "Axes | None" = None,
    frame: str = "sample",
) -> "tuple[Figure, Axes]":
    """Plot a 3D parallel acquisition and volume in XYZ coordinates.

    Args:
        projection_geometry (Parallel3DGeometry): Scan to display. Tomography,
            laminography and explicit vectors are supported. Effective detector
            centres include detector shifts and centre-of-rotation corrections.
        volume_geometry (Volume3DGeometry): Centred, unit-spaced working volume.
            Its (Z, Y, X) shape is converted to XYZ spatial bounds.

    Keyword Args:
        frame (str): "sample" (default) keeps the volume fixed and shows the
            acquisition rotating. "laboratory" keeps the beam and detector
            orientation fixed and rotates the tilted sample. Laboratory mode
            requires tomography or an untransformed laminography() trajectory.
            The detector is displayed downstream at a schematic distance;
            neither geometry nor projection data are modified. A curved arrow
            shows sample rotation inferred from acquisition order (opposite to
            acquisition-frame angles). The arc is a schematic symbol attached
            to the axis tip, kept readable at edge-on observer angles.
            Single-view or reversing trajectories
            have no single rotation direction and omit this arrow.
        view_indices (sequence[int] or None): Distinct zero-based views in the
            requested display order. Default None selects up to three evenly
            spaced acquisition indices. Empty, negative and invalid indices
            are rejected. These are acquisition indices, not angle values.
        ax (matplotlib.axes.Axes or None): Existing 3D axes, default None creates
            a figure and axes. Existing artists are retained. Limits are reset
            to frame this geometry, using equal spatial scaling on all axes.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: Figure and 3D axes
        for further customisation or saving. This function does not call show().

    Notes:
        Matplotlib is optional; install tomobar[visualisation] to plot. Neither
        ASTRA nor CuPy is imported, and no CUDA device or projector is needed.
        Detector outlines follow outer pixel edges and preserve non-unit or
        oblique pixel basis vectors. Arrows show each central ray's direction;
        their finite display lengths do not represent source distances.

        In sample mode detectors are drawn at their supplied reference planes. In parallel
        geometry these planes may intersect the volume; they are not moved to
        an artificial source/detector distance. A rotation axis cannot in
        general be inferred from vector geometry, so only XYZ axes are labelled.
    """
    if not isinstance(projection_geometry, Parallel3DGeometry):
        raise TypeError("projection_geometry must be a Parallel3DGeometry")
    if not isinstance(volume_geometry, Volume3DGeometry):
        raise TypeError("volume_geometry must be a Volume3DGeometry")
    if frame not in ("sample", "laboratory"):
        raise ValueError("frame must be 'sample' or 'laboratory'")
    if frame == "laboratory" and (
        projection_geometry.angles is None and projection_geometry._laminography is None
    ):
        raise ValueError(
            "laboratory frame requires tomography or a laminography() trajectory"
        )
    count = projection_geometry.view_count
    if view_indices is None:
        indices = np.linspace(0, count - 1, min(3, count), dtype=int)
    else:
        indices = np.asarray(view_indices)
        if indices.ndim != 1 or not indices.size:
            raise ValueError("view_indices must be a nonempty one-dimensional sequence")
        if (
            any(isinstance(index, (bool, np.bool_)) for index in view_indices)
            or indices.dtype.kind not in "iu"
        ):
            raise TypeError("view_indices must contain integers, excluding booleans")
        if np.any(indices < 0) or np.any(indices >= count):
            raise ValueError("view_indices contains an out-of-range acquisition index")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("view_indices must contain distinct indices")

    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
    except ImportError as error:
        raise ImportError(
            "Geometry visualisation requires Matplotlib; install tomobar[visualisation]"
        ) from error
    if ax is None:
        figure = plt.figure(figsize=(8, 7))
        ax = figure.add_subplot(111, projection="3d")
    elif not isinstance(ax, Axes3D):
        raise TypeError("ax must be a Matplotlib 3D axes")
    else:
        figure = ax.figure

    if frame == "laboratory":
        return _plot_laboratory(
            projection_geometry, volume_geometry, indices, figure, ax
        )

    half_size = np.asarray(volume_geometry.shape[::-1], dtype=float) / 2
    corners = np.asarray(list(product((-1, 1), repeat=3))) * half_size
    edges = [
        corners[[i, j]]
        for i in range(8)
        for j in range(i + 1, 8)
        if np.count_nonzero(corners[i] != corners[j]) == 1
    ]
    ax.add_collection3d(Line3DCollection(edges, colors="black", linewidths=1.5))
    legend = [Line2D([], [], color="black", label="Volume")]
    bounds = [corners]
    ray_length = float(2 * np.linalg.norm(half_size))
    rows, columns = projection_geometry.detector_shape
    # to_vectors applies shifts and CoR once, in the supplied detector basis.
    vectors = projection_geometry.to_vectors()[indices]
    for position, (index, vector) in enumerate(zip(indices, vectors)):
        ray, centre, u, v = vector.reshape(4, 3)
        horizontal, vertical = columns * u / 2, rows * v / 2
        detector = np.asarray(
            [
                centre - horizontal - vertical,
                centre + horizontal - vertical,
                centre + horizontal + vertical,
                centre - horizontal + vertical,
            ]
        )
        colour = plt.get_cmap("tab10")(position % 10)
        ax.add_collection3d(
            Poly3DCollection(
                [detector],
                facecolors=[colour],
                edgecolors=[colour],
                alpha=0.2,
            )
        )
        # Scale before normalising so very large or small vector magnitudes
        # do not overflow or underflow while choosing the display direction.
        direction = ray / np.max(np.abs(ray))
        direction /= np.linalg.norm(direction)
        start = centre - ray_length * direction / 2
        end = centre + ray_length * direction / 2
        ax.quiver(
            *start, *direction, length=ray_length, color=colour, arrow_length_ratio=0.08
        )
        legend.append(Line2D([], [], color=colour, label=f"View {index}"))
        bounds.extend((detector, np.asarray([start, end])))

    points = np.concatenate(bounds)
    centre = (points.min(axis=0) + points.max(axis=0)) / 2
    radius = float(np.max(np.ptp(points, axis=0))) * 0.55
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_zlim(centre[2] - radius, centre[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X (voxel units)")
    ax.set_ylabel("Y (voxel units)")
    ax.set_zlabel("Z (voxel units)")
    ax.set_title("Volume, detector planes and central ray directions")
    ax.legend(handles=legend, loc="upper left")
    return figure, ax
