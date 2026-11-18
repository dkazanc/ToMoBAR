"""Generate projections, reconstruct a small sample and save a comparison."""

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
from tomobar.reconstruction.direct import RecToolsDIR

# A known sample in (z, y, x) order, with unit voxel and detector spacing.
size = 48
coordinates = np.arange(size, dtype=np.float32) - (size - 1) / 2
z, y, x = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
phantom = np.zeros((size, size, size), dtype=np.float32)
phantom[(x / 15) ** 2 + (y / 12) ** 2 + (z / 16) ** 2 <= 1] = 0.5
phantom[((x - 7) / 5) ** 2 + ((y + 3) / 6) ** 2 + (z / 8) ** 2 <= 1] = 1.0

# Half-turn parallel-beam acquisition; exclude the duplicate endpoint.
scan = Parallel3DGeometry.tomography(
    angles=np.linspace(0, np.pi, 90, endpoint=False),
    detector_shape=(size, int(np.ceil(np.sqrt(2) * size))),
)
grid = Volume3DGeometry(shape=phantom.shape)

with RecToolsDIR(projection_geometry=scan, volume_geometry=grid, device_index=0) as rec:
    projections = rec.forward(phantom)  # (detY, angles, detX)
    reconstructed = cp.asnumpy(
        rec.FBP(projections, filter_type="RAM-LAK", recon_mask_radius=None)
    )

print(f"Projection shape (detY, angles, detX): {projections.shape}")
print(f"Reconstruction shape (z, y, x): {reconstructed.shape}")
print(f"FBP RMSE: {np.sqrt(np.mean((reconstructed - phantom) ** 2)):.5f}")

figure, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
for axis, image, title in zip(
    axes, (phantom, reconstructed), ("Known sample", "FBP reconstruction")
):
    axis.imshow(image[size // 2], origin="lower", cmap="gray", vmin=0, vmax=1)
    axis.set_title(title)
    axis.set_axis_off()
figure.savefig("first_reconstruction.png", dpi=150)
plt.close(figure)
print("Saved first_reconstruction.png")
