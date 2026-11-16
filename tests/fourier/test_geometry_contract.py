"""CPU references for Fourier detector phases and geometry restrictions."""

from dataclasses import replace

import numpy as np
import pytest

from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
from tomobar.fourier_plan import build_fourier_plan


@pytest.mark.parametrize("offset", [-1.25, 0, 0.5])
def test_fourier_inversion_accepts_cor_but_not_shifts(offset):
    scan = Parallel3DGeometry(
        angles=[0, 1, 2], detector_shape=(2, 8), centre_of_rotation=offset
    )
    grid = Volume3DGeometry((2, 8, 8))
    plan = build_fourier_plan(scan, grid, scan.projection_shape)
    assert plan is not None
    for shifts in (np.ones((3, 2)), [[0, 0], [0, 0.25], [0, 0]]):
        shifted_scan = replace(scan, detector_shifts=shifts)
        with pytest.raises(ValueError, match="detector_shifts"):
            build_fourier_plan(shifted_scan, grid, scan.projection_shape)
