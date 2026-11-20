#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualise tomography, laminography, shifts and explicit parallel geometry.

This cell-based example requires NumPy and Matplotlib, but no ASTRA or CUDA.
Detector planes are displayed downstream at schematic distances, and beam
arrows end at their centres. Use detector_position="reference" to inspect the
original reference planes. Arrows indicate parallel ray directions, not source distance.
Known trajectories show a rotation axis and acquisition-rotation arc. The final
comparison uses a fixed instrument and shows the opposite sample rotation.
"""

import numpy as np
import matplotlib.pyplot as plt
from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
from tomobar.visualisation import plot_geometry

# %%
# Volume arrays use (Z, Y, X); plots use physical XYZ coordinates.
reconstruction_grid = Volume3DGeometry(shape=(24, 32, 48))
angles_rad = np.linspace(0, np.pi, 36, endpoint=False)
detector_shape = (32, 64)
view_indices = [0, 12, 24]

# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%% Tomography and laminography geometry %%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

tomography = Parallel3DGeometry.tomography(
    angles=angles_rad, detector_shape=detector_shape, centre_of_rotation=0.0
)
laminography = Parallel3DGeometry.laminography(
    angles=angles_rad, detector_shape=detector_shape, tilt_angle=np.pi / 3
)

fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(121, projection="3d")
plot_geometry(tomography, reconstruction_grid, view_indices=view_indices, ax=ax)
ax.set_title("Standard tomography\nDetectors downstream (schematic)")
ax = fig.add_subplot(122, projection="3d")
plot_geometry(laminography, reconstruction_grid, view_indices=view_indices, ax=ax)
ax.set_title("Laminography: 60-degree tilt\nDetectors downstream (schematic)")
plt.tight_layout()
plt.show()
# fig.savefig('geometry_tomography_laminography.png', dpi=200)

# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%% Detector shifts and explicit vector geometry %%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

# Horizontal/vertical shifts and CoR are in detector pixels, applied once.
shifted_scan = Parallel3DGeometry.tomography(
    angles=angles_rad,
    detector_shape=detector_shape,
    detector_shifts=np.tile([6.0, -4.0], (len(angles_rad), 1)),
    centre_of_rotation=3.0,
)

# Explicit vectors can represent tilted detectors and non-unit pixel sizes.
vectors = laminography.to_vectors()
vectors[:, 6:9] *= 1.5  # column step: 1.5 voxel units
vectors[:, 9:12] *= 0.75  # row step: 0.75 voxel units
vector_scan = Parallel3DGeometry.from_vectors(
    vectors=vectors, detector_shape=detector_shape
)

fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(121, projection="3d")
plot_geometry(shifted_scan, reconstruction_grid, view_indices=view_indices, ax=ax)
ax.set_title("Tomography: detector shifts and CoR\nDetectors downstream (schematic)")
ax = fig.add_subplot(122, projection="3d")
plot_geometry(vector_scan, reconstruction_grid, view_indices=view_indices, ax=ax)
ax.set_title(
    "Explicit vectors: non-unit detector pixels\nDetectors downstream (schematic)"
)
plt.tight_layout()
plt.show()
# fig.savefig('geometry_shifts_vectors.png', dpi=200)
# %%


# %%
# Laboratory frame: incoming beam and downstream detector stay fixed.
fig = plt.figure(figsize=(14, 6))
for position, scan, title in [
    (1, tomography, "Tomography: fixed instrument"),
    (2, laminography, "Laminography: fixed instrument"),
]:
    ax = fig.add_subplot(1, 2, position, projection="3d")
    plot_geometry(
        scan, reconstruction_grid, view_indices=[0], ax=ax, frame="laboratory"
    )
    ax.set_title(title)
plt.tight_layout()
plt.show()
# fig.savefig('geometry_laboratory.png', dpi=200)
