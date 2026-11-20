"""Laboratory-frame geometry plotting helpers."""

from itertools import product
import numpy as np


def _axis_rotation(axis, angle):
    """Right-handed Rodrigues rotation, without a SciPy dependency."""
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return (
        np.cos(angle) * np.eye(3)
        + np.sin(angle) * cross
        + (1 - np.cos(angle)) * np.outer(axis, axis)
    )


def _laboratory_transforms(scan, indices):
    """Map sample coordinates to a fixed beam (-Y), detector columns X/rows Z."""
    trajectory = scan._laminography
    if trajectory is None:
        angles, axis = scan.angles, np.array([0.0, 0.0, 1.0])
        alignment = np.eye(3)
    else:
        angles = trajectory.angles
        ray, _, u, v = trajectory.vectors[0].reshape(4, 3)
        tilt = trajectory.tilt_angle
        axis = np.cos(tilt) * ray + np.sin(tilt) * v
        axis /= np.linalg.norm(axis)
        undo_first = _axis_rotation(axis, -angles[0])
        alignment = np.array([undo_first @ u, -(undo_first @ ray), undo_first @ v])
    return [
        alignment @ _axis_rotation(axis, -angles[i]) for i in indices
    ], alignment @ axis


def _sample_rotation_arc(scan, rotation, axis, scale):
    """Arc normal to the sample axis, signed by acquisition order."""
    angles = scan.angles if scan._laminography is None else scan._laminography.angles
    steps = np.diff(np.unwrap(angles))
    moving = steps[np.abs(steps) > 1e-12]
    if not moving.size or not (np.all(moving > 0) or np.all(moving < 0)):
        return None
    # Sample motion is the inverse of the acquisition-frame rotation.
    sign = -np.sign(moving[0])
    phase = np.deg2rad(20) + sign * np.linspace(0, np.deg2rad(260), 100)
    # Choose the sample basis least aligned with the axis, then project it
    # into the rotation plane. Sample X alone degenerates for X-aligned axes.
    basis = rotation[:, np.argmin(np.abs(rotation.T @ axis))]
    e1 = basis - np.dot(basis, axis) * axis
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    return 0.35 * scale * axis + 0.24 * scale * (
        np.cos(phase)[:, None] * e1 + np.sin(phase)[:, None] * e2
    )


def _plot_laboratory(scan, volume, indices, figure, ax, detector_position="downstream"):
    """Stationary instrument with tilted, rotating sample; distances schematic."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    rotations, axis = _laboratory_transforms(scan, indices)
    half = np.asarray(volume.shape[::-1], dtype=float) / 2
    corners = np.asarray(list(product((-1, 1), repeat=3))) * half
    edge_indices = [
        (i, j)
        for i in range(8)
        for j in range(i + 1, 8)
        if np.count_nonzero(corners[i] != corners[j]) == 1
    ]
    scale = float(max(volume.shape + scan.detector_shape))
    distance = 1.15 * scale if detector_position == "downstream" else 0.0
    bounds, legend = [], []
    vectors = scan.to_vectors()[indices]
    rows, columns = scan.detector_shape
    first_detector = None
    for position, (index, rotation, vector) in enumerate(
        zip(indices, rotations, vectors)
    ):
        colour = plt.get_cmap("tab10")(position % 10)
        sample = corners @ rotation.T
        ax.add_collection3d(
            Line3DCollection(
                [sample[[i, j]] for i, j in edge_indices],
                colors=[colour],
                linewidths=1.6,
            )
        )
        # A labelled sample-X arrow makes rotation visible even for square slabs.
        sample_x = rotation[:, 0]
        ax.quiver(
            0, 0, 0, *sample_x, length=half[0], color=colour, arrow_length_ratio=0.18
        )
        ax.text(*(half[0] * sample_x), "Sample X", color=colour)
        legend.append(Line2D([], [], color=colour, label=f"Sample at view {index}"))
        ray, centre, u, v = vector.reshape(4, 3) @ rotation.T
        direction = ray / np.linalg.norm(ray)
        centre = centre + distance * direction
        detector = np.array(
            [
                centre - columns * u / 2 - rows * v / 2,
                centre + columns * u / 2 - rows * v / 2,
                centre + columns * u / 2 + rows * v / 2,
                centre - columns * u / 2 + rows * v / 2,
            ]
        )
        # Identical instrument poses need drawing only once. Per-view detector
        # shifts remain visible, rather than silently replacing them by zero.
        if first_detector is None or not np.allclose(detector, first_detector):
            ax.add_collection3d(
                Poly3DCollection(
                    [detector], facecolors="royalblue", edgecolors="navy", alpha=0.25
                )
            )
            ax.scatter(*centre, color="navy", s=20)
            ax.text(
                *(centre + rows * v / 2 + 0.10 * scale * v), "Detector", color="navy"
            )
            for basis, label in [(u, "u"), (v, "v")]:
                ax.quiver(
                    *centre,
                    *basis,
                    length=0.15 * scale,
                    color="navy",
                    arrow_length_ratio=0.2,
                )
                ax.text(*(centre + 0.18 * scale * basis), label, color="navy")
            for offset in [-0.22 * columns, 0, 0.22 * columns]:
                start = centre - (distance + 0.85 * scale) * direction + offset * u
                ax.quiver(
                    *start,
                    *direction,
                    length=distance + 0.85 * scale,
                    color="darkorange",
                    arrow_length_ratio=0.08,
                    linewidth=1.6,
                )
                middle = start + 0.35 * scale * direction
                ax.quiver(
                    *middle,
                    *direction,
                    length=0.3 * scale,
                    color="darkorange",
                    arrow_length_ratio=0.35,
                )
                bounds.append(np.array([start, centre + offset * u]))
            ax.text(
                *(centre - (distance + 0.95 * scale) * direction),
                "Incoming beam",
                color="darkorange",
            )
        if first_detector is None:
            first_detector = detector
        bounds.extend([sample, detector])
    axis_ends = np.array([-0.6 * scale * axis, 0.6 * scale * axis])
    ax.plot(*axis_ends.T, "k--", linewidth=1.5)
    ax.text(*(1.22 * axis_ends[1]), "Rotation axis")
    bounds.append(axis_ends)
    arc = _sample_rotation_arc(scan, rotations[0], axis, scale)
    if arc is not None:
        radial = arc - 0.35 * scale * axis
        sign = np.sign(np.dot(np.cross(radial[0], radial[1]), axis))
        _add_rotation_symbol(ax, axis_ends[1], axis, sign)
        legend.append(Line2D([], [], color="crimson", label="Sample rotation →"))
    points = np.concatenate(bounds)
    middle = (points.min(axis=0) + points.max(axis=0)) / 2
    radius = float(np.max(np.ptp(points, axis=0))) * 0.65
    ax.set(
        xlim=(middle[0] - radius, middle[0] + radius),
        ylim=(middle[1] - radius, middle[1] + radius),
        zlim=(middle[2] - radius, middle[2] + radius),
        xlabel="Lab X (voxel units)",
        ylabel="Lab Y (voxel units)",
        zlabel="Lab Z (voxel units)",
    )
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=23, azim=-55)
    ax.set_title(
        "Fixed beam and detector; rotating sample"
        + (
            "\nDetector distance schematic"
            if detector_position == "downstream"
            else "\nDetector reference planes"
        )
    )
    legend.extend(
        [
            Line2D([], [], color="darkorange", label="Beam → detector"),
            Line2D([], [], color="royalblue", label="Detector"),
            Line2D([], [], color="black", linestyle="--", label="Rotation axis"),
        ]
    )
    ax.legend(handles=legend, loc="upper left", fontsize="small")
    return figure, ax


def _add_rotation_symbol(ax, position, axis, sign):
    """Attach a readable schematic arc to the projected top of the axis.

    Unlike a physical ring, this annotation keeps an elliptical opening even
    when the rotation plane is viewed edge-on. Its centre remains attached to
    the 3D axis as the observer rotates the plot; sign is viewed from +axis.
    """
    from matplotlib.artist import Artist
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path
    from mpl_toolkits.mplot3d import proj3d

    class RotationSymbol(Artist):
        def draw(self, renderer):
            if not self.get_visible():
                return
            projection = ax.get_proj()

            def screen(point):
                x, y, _ = proj3d.proj_transform(*point, projection)
                return ax.transData.transform((x, y))

            centre = screen(position)
            up = screen(position + axis) - centre
            norm = np.linalg.norm(up)
            up = up / norm if norm > 1e-12 else np.array([0.0, 1.0])
            right = np.array([up[1], -up[0]])
            phase = np.deg2rad(25) + sign * np.linspace(0, np.deg2rad(285), 100)
            size = renderer.points_to_pixels(25)
            vertices = (
                centre
                + size * np.cos(phase)[:, None] * right
                + 0.40 * size * np.sin(phase)[:, None] * up
            )
            patch = FancyArrowPatch(
                path=Path(vertices),
                arrowstyle="-|>",
                mutation_scale=14,
                color="crimson",
                linewidth=2,
                transform=None,
                clip_on=False,
            )
            patch.set_figure(ax.figure)
            patch.draw(renderer)
            self.stale = False

    symbol = RotationSymbol()
    symbol.set_label("Rotation arrow at axis tip")
    symbol.set_zorder(10)
    ax.add_artist(symbol)
    return symbol
