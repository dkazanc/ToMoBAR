"""Dataset-specific APS angle repair and intensity preparation checks."""

from pathlib import Path
import runpy

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

demo = runpy.run_path(
    str(
        Path(__file__).resolve().parents[2]
        / "demos/experimental/laminography/_aps_leaf_preprocessing.py"
    )
)


def test_missing_view_uses_angle_neighbours_not_frame_indices():
    angles, left, right, fractions = demo["uniform_view_plan"]([0, 90, 270, 360], 1)
    np.testing.assert_array_equal(angles, [0, 90, 180, 270, 360])
    np.testing.assert_array_equal(left, [0, 1, 1, 2, 3])
    np.testing.assert_array_equal(right, [0, 1, 2, 2, 3])
    np.testing.assert_array_equal(fractions, [0, 0, 0.5, 0, 0])


@pytest.mark.parametrize(
    "angles,stride",
    [([0, 90, 270, 360], 3), ([0, 90, 91, 360], 1), ([0, 90, 90, 360], 1)],
)
def test_rejects_invalid_angular_sampling(angles, stride):
    with pytest.raises(ValueError):
        demo["uniform_view_plan"](angles, stride)


def test_binning_before_log_and_missing_view_interpolation(tmp_path):
    path = tmp_path / "leaf.h5"
    # Alternating detector columns make averaging intensity before the log
    # distinguishable from averaging logarithms.
    raw = np.broadcast_to([4, 8] * 4, (4, 4, 8)).copy().astype(np.uint16)
    raw[1] = 4
    raw[2] = 3
    with h5py.File(path, "w") as source:
        source["exchange/data"] = raw
        source["exchange/data_white"] = np.full((2, 4, 8), 10, dtype=np.uint16)
        source["exchange/data_dark"] = np.full((2, 4, 8), 2, dtype=np.uint16)
        source["exchange/theta"] = [0, 90, 270, 360]
        source["measurement/instrument/detection_system/objective/resolution"] = [1.4]
    data, angles, metadata = demo["prepare_projections"](path, 2, 1)
    assert data.shape == (2, 5, 4)
    assert data.dtype == np.float32
    expected = np.log([2, 4, np.sqrt(32), 8, 2])
    np.testing.assert_allclose(
        data, np.broadcast_to(expected[None, :, None], data.shape)
    )
    assert metadata["interpolated_angles_degrees"] == [180]
    assert metadata["voxel_size_um"] == 2.8
    # Uniform subsampling keeps the whole turn, including its endpoint.
    _, sampled_angles, _ = demo["prepare_projections"](path, 2, 2)
    np.testing.assert_array_equal(sampled_angles, angles[::2])


def test_binning_retains_detector_field_of_view():
    raw = np.arange(32, dtype=np.uint16).reshape(1, 4, 8)
    binned = demo["read_binned_frame"](raw, 0, 2)
    np.testing.assert_array_equal(
        binned, [[4.5, 6.5, 8.5, 10.5], [20.5, 22.5, 24.5, 26.5]]
    )


def test_missing_dataset_explains_download(tmp_path):
    missing = tmp_path / "leaf_lam_20deg_roi1_417.h5"
    with pytest.raises(
        FileNotFoundError, match="Download leaf_lam_20deg_roi1_417.h5"
    ) as error:
        demo["prepare_projections"](str(missing), 8, 1)
    assert "tomo_00100" in str(error.value)
    assert "Globus" in str(error.value)
    assert "data_path" in str(error.value)
