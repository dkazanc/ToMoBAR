"""Input and lifecycle contracts for the standalone USFFT numerical backend."""

import numpy as np
import pytest


@pytest.fixture
def backend_type():
    pytest.importorskip("cupy")
    from tomobar._usfft_backend import USFFTBackend

    return USFFTBackend


def test_subset_indices_and_input_copy(backend_type, monkeypatch):
    monkeypatch.setattr(
        backend_type,
        "_initialize",
        lambda self, n, theta, *args: setattr(self, "angles", theta),
    )
    angles = np.linspace(0, np.pi, 7)
    backend = backend_type(size=8, detector_width=9, angles=angles, subset_count=3)
    angles[:] = 0
    np.testing.assert_array_equal(backend.subset_indices[0], [0, 3, 6])
    np.testing.assert_array_equal(backend.subset_indices[1], [1, 4])
    np.testing.assert_array_equal(backend._selected_angles(2), backend.angles[[2, 5]])
    assert backend.angles.dtype == np.float32
    assert backend.angles[-1] != 0
    with pytest.raises(TypeError):
        backend._selected_angles(True)
    with pytest.raises(ValueError):
        backend._selected_angles(3)
    backend.close()
    backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        backend._selected_angles(None)


@pytest.mark.parametrize(
    "options",
    [
        {"size": 0},
        {"size": True},
        {"detector_width": 3},
        {"angles": []},
        {"angles": [np.nan]},
        {"angles": [[1]]},
        {"mask_radius_squared": 0},
        {"centre_offset": np.inf},
        {"subset_count": 4},
        {"device_index": -1},
    ],
)
def test_invalid_configuration_before_cuda(backend_type, options):
    arguments = dict(size=4, detector_width=4, angles=[0, 1, 2])
    arguments.update(options)
    with pytest.raises((ValueError, TypeError)):
        backend_type(**arguments)


@pytest.mark.parametrize(
    "size,rows,width", [(8, 2, 8), (8, 3, 9), (9, 3, 10), (96, 2, 96)]
)
def test_gpu_execution_and_subsets(gpu, backend_type, size, rows, width):
    angles = np.linspace(0, np.pi, 7, endpoint=False)
    rng = np.random.default_rng(4)
    volume = rng.random((rows, size, size), dtype=np.float32)
    with backend_type(
        size=size, detector_width=width, angles=angles, subset_count=3
    ) as backend:
        projected = backend.forward(volume)
        assert projected.shape == (rows, 7, width)
        assert projected.dtype == np.float32
        assert np.isfinite(projected).all()
        assert np.any(projected)
        for index, views in enumerate(backend.subset_indices):
            subset = backend.forward(volume, subset_index=index)
            np.testing.assert_allclose(
                subset, projected[:, views], rtol=1e-5, atol=1e-6
            )
            reconstructed = backend.backproject(subset, subset_index=index)
            assert reconstructed.shape == volume.shape
            assert np.isfinite(reconstructed).all()
        # A strided input exercises packing before raw CUDA pointer access.
        device_result = backend.backproject(gpu.asarray(projected[:, :, ::-1]))
        host_result = backend.backproject(projected[:, :, ::-1])
        assert isinstance(device_result, gpu.ndarray)
        gpu.testing.assert_allclose(
            device_result, gpu.asarray(host_result), rtol=1e-5, atol=1e-6
        )


@pytest.mark.parametrize("size", [8, 9])
@pytest.mark.parametrize("offset", [-1.25, 0, 0.375])
def test_cor_passed_without_implicit_half_pixel(
    backend_type, monkeypatch, size, offset
):
    received = []
    monkeypatch.setattr(
        backend_type,
        "_initialize",
        lambda self, n, theta, mask: received.append(self.centre_offset),
    )
    backend = backend_type(
        size=size, detector_width=size, angles=[0, 1], centre_offset=offset
    )
    assert backend.centre_offset == offset
    assert received == [offset]
    backend.close()


@pytest.mark.parametrize("offset", [None, True, [0.25], 1j, float("nan")])
def test_invalid_scalar_cor_before_cuda(backend_type, offset):
    with pytest.raises((ValueError, TypeError)):
        backend_type(size=8, detector_width=8, angles=[0, 1], centre_offset=offset)
