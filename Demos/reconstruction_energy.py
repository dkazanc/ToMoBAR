#!/usr/bin/env python3
"""Return and plot reconstruction energy using the usual method calls.

Both methods reconstruct the same Gaussian-noisy projection data.
Requires ASTRA/CuPy with CUDA. No external dataset or TomoPhantom is needed.
"""

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
from tomobar.methodsIR import RecToolsIR
from tomobar.options import DataFidelity, Regularisation

# %% Create a small phantom and matching parallel-beam geometry.
shape = (8, 48, 48)
phantom = cp.zeros(shape, dtype=cp.float32)
phantom[:, 12:36, 14:34] = 0.02
phantom[:, 20:28, 20:28] = 0.01

scan = Parallel3DGeometry(
    angles=np.linspace(0, np.pi, 60, endpoint=False),
    detector_shape=(shape[0], 72),
)
grid = Volume3DGeometry(shape=shape)
fidelity = DataFidelity(method="LS")
regularisation = Regularisation(method="PD_TV", strength=0.01, iterations=50)

# %% Enable return_energy to receive (reconstruction, energy_vector).
# The ordinary image-only return remains the default when it is omitted.
with RecToolsIR(projection_geometry=scan, volume_geometry=grid) as rec:
    projections = rec.forward(phantom)

    # Gaussian measurement noise: standard deviation is 2% of the peak signal.
    noise_fraction = 0.02  # Adjust this to change the noise level.
    rng = np.random.default_rng(42)
    noise_std = noise_fraction * float(projections.max())
    noise = rng.normal(0.0, noise_std, size=projections.shape).astype(np.float32)
    projections += cp.asarray(noise)

    reconstruction_fista, energy_fista = rec.FISTA(
        projections,
        fidelity=fidelity,
        regularisation=regularisation,
        nonnegativity=True,
        iterations=50,
        recon_mask_radius=None,
        return_energy=True,
    )

    reconstruction_admm, energy_admm = rec.ADMM(
        projections,
        fidelity=fidelity,
        regularisation=regularisation,
        data_solver="cg",
        cg_iterations=5,
        nonnegativity=True,
        iterations=50,
        recon_mask_radius=None,
        return_energy=True,
    )

# %% Compare reconstructed slices using the same intensity range.
slice_index = shape[0] // 2
plt.figure(figsize=(12, 4))
for index, (title, image) in enumerate(
    [
        ("Phantom", phantom),
        ("FISTA-LS-TV", reconstruction_fista),
        ("ADMM-CG-LS-TV", reconstruction_admm),
    ],
    start=1,
):
    plt.subplot(1, 3, index)
    plt.imshow(cp.asnumpy(image[slice_index]), cmap="gray", vmin=0, vmax=0.02)
    plt.title(title)
    plt.axis("off")
plt.tight_layout()

# %% Energy is a NumPy vector: the initial value followed by completed iterations.
# Both methods use the same LS + strength*TV objective here. One ADMM iteration
# includes inner CG steps, so iteration counts are not a comparison of runtime.
plt.figure()
plt.plot(energy_fista, label="FISTA-LS-TV")
plt.plot(energy_admm, label="ADMM-CG-LS-TV")
plt.xlabel("Iteration (0 = initial image)")
plt.ylabel("Energy")
plt.title("Reconstruction energy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
