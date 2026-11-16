#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualise tomography, laminography, shifts and explicit parallel geometry.

This cell-based example requires NumPy and Matplotlib, but no ASTRA or CUDA.
Detector reference planes are shown at their actual positions and may cross
through the volume. Arrows indicate parallel ray directions, not source distance.
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
ax.set_title("Standard tomography")
ax = fig.add_subplot(122, projection="3d")
plot_geometry(laminography, reconstruction_grid, view_indices=view_indices, ax=ax)
ax.set_title("Laminography: 60-degree tilt")
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
ax.set_title("Tomography: detector shifts and CoR")
ax = fig.add_subplot(122, projection="3d")
plot_geometry(vector_scan, reconstruction_grid, view_indices=view_indices, ax=ax)
ax.set_title("Explicit vectors: non-unit detector pixels")
plt.tight_layout()
plt.show()
# fig.savefig('geometry_shifts_vectors.png', dpi=200)
# %%
