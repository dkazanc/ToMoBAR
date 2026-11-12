"""Synthetic FISTA continuation and supplied geometry correction (requires CUDA).

This demonstrates state reuse, not automatic alignment estimation.
"""

from dataclasses import replace

import cupy as cp
import numpy as np

from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
from tomobar.methodsIR import RecToolsIR


def main():
    scan = Parallel3DGeometry(
        angles=np.linspace(0, np.pi, 45, endpoint=False), detector_shape=(8, 48)
    )
    grid = Volume3DGeometry((8, 32, 32))
    phantom = cp.zeros(grid.shape, dtype=cp.float32)
    phantom[:, 8:24, 10:22] = 1
    with RecToolsIR(projection_geometry=scan, volume_geometry=grid) as rec:
        projections = rec.forward(phantom)
        # Deliberately start with an incorrect horizontal offset.
        rec.replace_geometry(replace(scan, centre_of_rotation=1.0))
        state = rec.initialise_solver(
            "FISTA", projections, return_energy=True, nonnegativity=True
        )
        rec.run(state, iterations=5)
        rec.run(state, iterations=5)  # retains FISTA momentum
        # Supply the known correction; a future alignment estimator would
        # generate this geometry from measured and predicted projections.
        rec.replace_geometry(scan)
        rec.run(state, iterations=10)  # restarts momentum and norm estimate
        image = rec.solver_image(state, recon_mask_radius=None)
        print("Output shape:", image.shape)
        print("Iterations:", state.iteration, "geometry restarts:", state.restart_count)
        print("Energy:", state.energy)


if __name__ == "__main__":
    main()
