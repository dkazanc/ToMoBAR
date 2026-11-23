"""Frame-wise preparation of the APS leaf laminography dataset."""

from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np

DATASET_URL = (
    "https://tomobank.readthedocs.io/en/latest/source/data/"
    "docs.data.laminography.html#camera-caffe"
)


def require_data_file(data_file):
    """Require the user to download the acquisition before running the demo."""
    path = Path(data_file).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"APS leaf dataset not found: {path}\n"
            "Download leaf_lam_20deg_roi1_417.h5 first using the tomo_00100 "
            f"Globus download link at {DATASET_URL}\n"
            "Then set data_path in aps_leaf_laminography.py to the downloaded file. "
            "The acquisition is approximately 29 GiB; it is not downloaded automatically."
        )
    return path


def uniform_view_plan(theta, view_step):
    """Map the recorded full turn to a uniform grid, filling isolated lost views."""
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim != 1 or len(theta) < 4 or not np.all(np.isfinite(theta)):
        raise ValueError("exchange/theta must contain finite angles in degrees")
    differences = np.diff(theta)
    if np.any(differences <= 0) or not np.isclose(theta[-1] - theta[0], 360):
        raise ValueError(
            "Expected increasing angles including both full-turn endpoints"
        )
    intervals = int(round(360 / np.median(differences)))
    if view_step < 1 or intervals % view_step:
        raise ValueError(f"view-step must be a positive divisor of {intervals}")
    step = 360 / intervals
    recorded = np.rint((theta - theta[0]) / step).astype(int)
    if not np.allclose(theta, theta[0] + recorded * step, atol=5e-4, rtol=0):
        raise ValueError("Recorded angles do not lie on a uniform acquisition grid")
    gaps = np.diff(recorded)
    if np.any((gaps < 1) | (gaps > 2)):
        raise ValueError("Only isolated missing views can be interpolated by this demo")
    targets = np.arange(0, intervals + 1, view_step)
    right = np.searchsorted(recorded, targets)
    left = np.where(recorded[right] == targets, right, right - 1)
    fractions = np.zeros(len(targets), dtype=np.float32)
    missing = left != right
    fractions[missing] = (targets[missing] - recorded[left[missing]]) / (
        recorded[right[missing]] - recorded[left[missing]]
    )
    return theta[0] + targets * step, left, right, fractions


def read_binned_frame(dataset, index, binning):
    """Average square detector blocks in intensity space, retaining the full FOV."""
    frame = np.asarray(dataset[int(index)], dtype=np.float32)
    rows, columns = frame.shape
    return frame.reshape(rows // binning, binning, columns // binning, binning).mean(
        axis=(1, 3)
    )


def prepare_projections(data_file, binning, view_step):
    """Read one frame at a time and interpolate missing log-corrected views."""
    data_file = require_data_file(data_file)
    with h5py.File(data_file, "r") as source:
        raw = source["exchange/data"]
        _, rows, columns = raw.shape
        if binning < 1 or rows % binning or columns % binning:
            raise ValueError(
                "binning must be positive and divide both detector dimensions"
            )
        theta = source["exchange/theta"][:]
        if len(theta) != raw.shape[0]:
            raise ValueError("Projection count and exchange/theta length differ")
        angles, left, right, fractions = uniform_view_plan(theta, view_step)

        def mean_frames(name):
            stack = source[f"exchange/{name}"]
            if not len(stack) or stack.shape[1:] != (rows, columns):
                raise ValueError(f"Invalid calibration stack: {name}")
            mean = np.zeros((rows // binning, columns // binning), dtype=np.float32)
            for index in range(len(stack)):
                mean += read_binned_frame(stack, index, binning) / len(stack)
            return mean

        flat = mean_frames("data_white")
        dark = mean_frames("data_dark")
        gain = flat - dark
        valid_gain = np.isfinite(gain) & (gain > 0)
        gain[~valid_gain] = 1
        print(
            f"Invalid flat/dark pixels after binning: {np.count_nonzero(~valid_gain)}"
        )

        @lru_cache(maxsize=2)
        def absorption(index):
            transmission = (read_binned_frame(raw, index, binning) - dark) / gain
            transmission[~valid_gain | ~np.isfinite(transmission)] = 1
            # Preserve phase-contrast excursions above one (negative absorption).
            return -np.log(np.maximum(transmission, np.float32(1e-6)))

        projections = np.empty(
            (rows // binning, len(angles), columns // binning), dtype=np.float32
        )
        for view, (lo, hi, fraction) in enumerate(zip(left, right, fractions)):
            frame = absorption(int(lo))
            if lo != hi:
                frame = (1 - fraction) * frame + fraction * absorption(int(hi))
                print(f"Interpolating missing view at {angles[view]:.2f} degrees")
            projections[:, view, :] = frame
            if view % 250 == 0:
                print(f"Prepared {view + 1}/{len(angles)} views", flush=True)

        pixel_path = "measurement/instrument/detection_system/objective/resolution"
        pixel_size_um = float(source[pixel_path][()].reshape(-1)[0])
        metadata = {
            "source": str(data_file.resolve()),
            "raw_shape": list(raw.shape),
            "pixel_size_um_from_file": pixel_size_um,
            "binning": binning,
            "voxel_size_um": pixel_size_um * binning,
            "projection_shape_detY_angles_detX": list(projections.shape),
            "interpolated_angles_degrees": angles[left != right].tolist(),
            "angles_degrees": angles.tolist(),
        }
    return projections, angles, metadata
