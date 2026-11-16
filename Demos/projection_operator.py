#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forward projection and unfiltered backprojection with ProjectionOperator.

Synthetic 3D CuPy and 2D NumPy examples, with no external data required.
Both examples require an ASTRA CUDA device. Raw backprojection is blurred
and has a different intensity scale from the phantom; it is not FBP.
"""

import timeit
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from tomobar.geometry import (
    Parallel2DGeometry,
    Parallel3DGeometry,
    Volume2DGeometry,
    Volume3DGeometry,
)
from tomobar.projectors.operator import ProjectionOperator

# %%
# Set the object size, detector dimensions and scanning geometry.
N_size = 64
detectorHoriz = 96
detectorVert = 8
angles_rad = np.linspace(0, np.pi, 90, endpoint=False)
subset_count = 3
device_index = 0
slice_index = detectorVert // 2

# Create a synthetic phantom containing two ellipses.
y, x = np.mgrid[:N_size, :N_size] - (N_size - 1) / 2
phantom_2D = (((x + 9) / 15) ** 2 + (y / 22) ** 2 < 1).astype(np.float32)
phantom_2D += 0.7 * (((x - 14) / 8) ** 2 + ((y - 5) / 11) ** 2 < 1)
phantom_3D = np.broadcast_to(phantom_2D, (detectorVert, N_size, N_size)).copy()

projection_geometry = Parallel3DGeometry(
    angles=angles_rad,
    detector_shape=(detectorVert, detectorHoriz),
    centre_of_rotation=0.0,
)
reconstruction_grid = Volume3DGeometry(shape=phantom_3D.shape)

# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%% Forward projection of a 3D CuPy volume %%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

# Initialise the operator once and reuse its cached projector for each call.
# Volume axes: (Z, Y, X). Projection axes: (detY, angles, detX).
with (
    cp.cuda.Device(device_index),
    ProjectionOperator(
        projection_geometry=projection_geometry,
        volume_geometry=reconstruction_grid,
        device_index=device_index,
        subset_count=subset_count,
    ) as Projector3D,
):
    phantom_cupy = cp.asarray(phantom_3D)
    tic = timeit.default_timer()
    projections_cupy = Projector3D.forward(phantom_cupy)
    cp.cuda.get_current_stream().synchronize()
    toc = timeit.default_timer()
    print("Forward projection done in {} seconds".format(toc - tic))
    print("Volume shape:", Projector3D.domain_shape)
    print("Projection shape:", Projector3D.range_shape)

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("%%%%%%%%%%%% Unfiltered 3D backprojection %%%%%%%%%%%%%%%%%%%")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    tic = timeit.default_timer()
    backprojection_cupy = Projector3D.backproject(projections_cupy)
    cp.cuda.get_current_stream().synchronize()
    toc = timeit.default_timer()
    print("Backprojection done in {} seconds".format(toc - tic))

    # Reuse caller-owned output buffers. Each call overwrites the supplied array.
    projection_buffer = cp.empty(Projector3D.range_shape, dtype=cp.float32)
    volume_buffer = cp.empty(Projector3D.domain_shape, dtype=cp.float32)
    Projector3D.forward(phantom_cupy, out=projection_buffer)
    Projector3D.backproject(projection_buffer, out=volume_buffer)

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("%%%%%%%%%%%% Projection and backprojection of a subset %%%%%%")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    # Subsets are interleaved in acquisition order. None selects all views.
    subset_index = min(1, Projector3D.subset_count - 1)
    view_indices = Projector3D.subset_indices[subset_index]
    subset_projections_cupy = cp.empty(
        Projector3D.subset_shape(subset_index), dtype=cp.float32
    )
    Projector3D.forward(
        phantom_cupy, subset_index=subset_index, out=subset_projections_cupy
    )
    subset_backprojection_cupy = Projector3D.backproject(
        subset_projections_cupy, subset_index=subset_index
    )
    print("Subset index:", subset_index, "number of views:", len(view_indices))
    print("Subset projection shape:", subset_projections_cupy.shape)
    print("Subset backprojection shape:", subset_backprojection_cupy.shape)

# The context releases the projector handles; the output arrays remain usable.
# %%
fig = plt.figure(figsize=(15, 4))
plt.subplot(131)
plt.imshow(phantom_3D[slice_index, :, :], cmap="gray")
plt.title("3D phantom: central slice")
plt.colorbar()
plt.subplot(132)
plt.imshow(cp.asnumpy(projections_cupy[slice_index, :, :]), cmap="gray", aspect="auto")
plt.title("Forward projection: central detector row")
plt.xlabel("Detector column")
plt.ylabel("View index")
plt.colorbar()
plt.subplot(133)
plt.imshow(cp.asnumpy(backprojection_cupy[slice_index, :, :]), cmap="gray")
plt.title("Unfiltered backprojection: central slice")
plt.colorbar()
plt.tight_layout()
plt.show()
# fig.savefig('projection_operator_3D.png', dpi=200)

# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%% Projection and backprojection with NumPy %%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

# NumPy input returns NumPy output. Projection still executes on the GPU.
# Volume axes: (Y, X). Projection axes: (angles, detX).
with ProjectionOperator(
    projection_geometry=Parallel2DGeometry(
        angles=angles_rad, detector_shape=(detectorHoriz,), centre_of_rotation=0.0
    ),
    volume_geometry=Volume2DGeometry(shape=phantom_2D.shape),
    device_index=device_index,
) as Projector2D:
    tic = timeit.default_timer()
    projections_numpy = Projector2D.forward(phantom_2D)
    backprojection_numpy = np.empty(Projector2D.domain_shape, dtype=np.float32)
    Projector2D.backproject(projections_numpy, out=backprojection_numpy)
    toc = timeit.default_timer()
    print("2D projection and backprojection done in {} seconds".format(toc - tic))

fig = plt.figure(figsize=(15, 4))
plt.subplot(131)
plt.imshow(phantom_2D, cmap="gray")
plt.title("2D phantom (NumPy)")
plt.colorbar()
plt.subplot(132)
plt.imshow(projections_numpy, cmap="gray", aspect="auto")
plt.title("Forward projection (NumPy)")
plt.xlabel("Detector column")
plt.ylabel("View index")
plt.colorbar()
plt.subplot(133)
plt.imshow(backprojection_numpy, cmap="gray")
plt.title("Unfiltered backprojection (NumPy)")
plt.colorbar()
plt.tight_layout()
plt.show()
# fig.savefig('projection_operator_2D.png', dpi=200)
# %%
