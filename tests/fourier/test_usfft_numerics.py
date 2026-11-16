"""Numerical contract, executing production FFTs with CPU gridding emulation."""

import numpy as np
import pytest

from tests.helpers.usfft_reference import (
    direct_fourier_projection,
    emulate_interpolation,
)


@pytest.fixture
def cpu_backend(monkeypatch):
    pytest.importorskip("cupy")
    import tomobar.projectors._usfft.backend as module

    monkeypatch.setattr(module, "cp", np)
    monkeypatch.setattr(module, "load_cuda_module", lambda *args: None)
    monkeypatch.setattr(module, "get_cuda_kernel", lambda *args: emulate_interpolation)

    def create(n, width, angles, cor=0, mask=1):
        backend = module.USFFTBackend.__new__(module.USFFTBackend)
        backend.size = n
        backend.detector_width = width
        backend.centre_offset = cor
        backend._workspace = None
        backend._view_plans = {}
        backend._initialize.__wrapped__(
            backend, n, np.asarray(angles, dtype=np.float32), mask
        )
        return backend

    return create


@pytest.mark.parametrize("n,rows,width", [(8, 1, 8), (9, 3, 9), (8, 3, 9), (9, 2, 12)])
@pytest.mark.parametrize("cor", [0, -1.25, 0.375])
def test_matches_direct_plane_wave_reference(cpu_backend, n, rows, width, cor):
    angles = np.array([0, np.pi / 2, -0.4, 0.73, np.pi, 4.1], dtype=np.float32)
    backend = cpu_backend(n, width, angles, cor)
    volume = np.random.default_rng(12).random((rows, n, n), dtype=np.float32)
    actual = backend._forward_impl(volume, angles)
    expected = direct_fourier_projection(volume, angles, width, cor)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    "n,rows,width", [(1, 1, 1), (8, 1, 8), (9, 3, 9), (8, 3, 9), (9, 2, 12)]
)
@pytest.mark.parametrize("cor", [0, -1.25, 0.375])
def test_adjoint_and_uneven_subsets(cpu_backend, n, rows, width, cor):
    angles = np.array([0.73, -0.4, np.pi / 2, 0, 4.1, 0.73, 2], dtype=np.float32)
    backend = cpu_backend(n, width, angles, cor)
    rng = np.random.default_rng(13)
    x = rng.normal(size=(rows, n, n)).astype(np.float32)
    y = rng.normal(size=(rows, len(angles), width)).astype(np.float32)
    full = backend._forward_impl(x, angles)
    total = np.zeros_like(x)
    for views in (np.arange(0, 7, 3), np.arange(1, 7, 3), np.arange(2, 7, 3)):
        ax = backend._forward_impl(x, angles[views])
        aty = backend._backproject_impl(y[:, views], angles[views])
        np.testing.assert_allclose(ax, full[:, views], rtol=2e-6, atol=2e-6)
        lhs = np.vdot(ax, y[:, views])
        rhs = np.vdot(x, aty)
        assert abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-6) < 2e-5
        total += aty
    np.testing.assert_allclose(
        total, backend._backproject_impl(y, angles), rtol=2e-5, atol=2e-5
    )


@pytest.mark.parametrize("n", [8, 9])
def test_cardinal_angles_orientation_mass_and_angle_count(cpu_backend, n):
    angles = np.array([0, np.pi / 2], dtype=np.float32)
    backend = cpu_backend(n, n, angles)
    volume = np.zeros((1, n, n), dtype=np.float32)
    volume[0, n // 2 - 1, n // 2 + 1] = 2.5
    projected = backend._forward_impl(volume, angles)
    np.testing.assert_allclose(projected[:, 0], volume.sum(axis=1), atol=2e-5)
    np.testing.assert_allclose(projected[:, 1], volume.sum(axis=2), atol=2e-5)
    np.testing.assert_allclose(projected.sum(axis=-1), 2.5, atol=2e-5)
    repeated = np.tile(angles, 3)
    other = cpu_backend(n, n, repeated)
    np.testing.assert_allclose(
        other._forward_impl(volume, repeated), np.tile(projected, (1, 3, 1)), atol=2e-5
    )


@pytest.mark.parametrize("n", [8, 9])
def test_integer_cor_and_padding_transpose(cpu_backend, n):
    angles = np.array([0.2, 0.7], dtype=np.float32)
    x = np.random.default_rng(3).normal(size=(1, n, n)).astype(np.float32)
    centred = cpu_backend(n, n + 2, angles)
    shifted = cpu_backend(n, n + 2, angles, 1)
    base = centred._forward_impl(x, angles)
    actual = shifted._forward_impl(x, angles)
    np.testing.assert_allclose(
        actual[..., 1:-1], np.roll(base[..., 1:-1], -1, axis=-1), atol=2e-5
    )
    assert not actual[..., [0, -1]].any()
    outside = np.zeros_like(base)
    outside[..., [0, -1]] = 1
    np.testing.assert_array_equal(shifted._backproject_impl(outside, angles), 0)


@pytest.mark.parametrize(
    "n,rows,width", [(8, 1, 8), (9, 3, 10), (32, 2, 35), (96, 3, 96)]
)
@pytest.mark.parametrize("cor", [0, -1.25, 0.375])
def test_cuda_numerical_contract(gpu, n, rows, width, cor):
    from tomobar.projectors._usfft.backend import USFFTBackend

    angles = np.array([0.73, -0.4, np.pi / 2, 0, 4.1, 0.73, 2], dtype=np.float32)
    rng = np.random.default_rng(19)
    host_x = rng.normal(size=(rows, n, n)).astype(np.float32)
    host_y = rng.normal(size=(rows, len(angles), width)).astype(np.float32)
    x, y = gpu.asarray(host_x), gpu.asarray(host_y)
    with USFFTBackend(
        size=n, detector_width=width, angles=angles, centre_offset=cor, subset_count=3
    ) as backend:
        ax = backend.forward(x)
        aty = backend.backproject(y)
        lhs, rhs = gpu.vdot(ax, y), gpu.vdot(x, aty)
        scale = gpu.linalg.norm(ax) * gpu.linalg.norm(y) + gpu.linalg.norm(
            x
        ) * gpu.linalg.norm(aty)
        assert float(abs(lhs - rhs) / scale) < 5e-6
        total = gpu.zeros_like(x)
        for subset, views in enumerate(backend.subset_indices):
            part = backend.forward(x, subset_index=subset)
            gpu.testing.assert_allclose(part, ax[:, views], rtol=3e-5, atol=3e-5)
            total += backend.backproject(y[:, views], subset_index=subset)
        gpu.testing.assert_allclose(total, aty, rtol=1e-4, atol=1e-4)
        if n <= 32:
            reference = direct_fourier_projection(host_x, angles, width, cor)
            np.testing.assert_allclose(gpu.asnumpy(ax), reference, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("n", [32, 33])
def test_cuda_cardinal_orientation_and_scale_against_astra(gpu, n):
    from tomobar.projectors.operator import ProjectionOperator
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry

    scan = Parallel3DGeometry(angles=[0, np.pi / 2], detector_shape=(3, n))
    grid = Volume3DGeometry((3, n, n))
    yy, xx = np.mgrid[:n, :n] - (n - 1) / 2
    phantom = np.exp(-((xx - 3) ** 2 + (yy + 2) ** 2) / 5)
    phantom += 0.3 * np.exp(-((xx + 4) ** 2 + (yy - 3) ** 2) / 3)
    phantom[xx**2 + yy**2 >= (n / 2) ** 2] = 0
    volume = gpu.asarray(
        np.stack([phantom, phantom * 2, phantom * 0.5]), dtype=gpu.float32
    )
    with ProjectionOperator(
        projection_geometry=scan, volume_geometry=grid, backend="astra"
    ) as astra:
        reference = astra.forward(volume)
    with ProjectionOperator(
        projection_geometry=scan, volume_geometry=grid, backend="usfft"
    ) as usfft:
        actual = usfft.forward(volume)
    # Different discretisations need not agree at general angles; cardinal
    # views of this smooth phantom check the shared physical units and axes.
    gpu.testing.assert_allclose(actual, reference, rtol=5e-3, atol=2e-3)


@pytest.mark.parametrize("n,width,cor", [(8, 8, 0), (8, 9, 0.375), (9, 12, -1.25)])
def test_backprojection_against_independent_dense_transpose(cpu_backend, n, width, cor):
    angles = np.array([0, 0.31, -0.8, 1.5], dtype=np.float32)
    basis_volumes = np.eye(n * n).reshape(n * n, n, n)
    matrix = (
        direct_fourier_projection(basis_volumes, angles, width, cor)
        .reshape(n * n, -1)
        .T
    )
    y = (
        np.random.default_rng(91)
        .normal(size=(1, len(angles), width))
        .astype(np.float32)
    )
    expected = (matrix.T @ y.ravel()).reshape(1, n, n)
    backend = cpu_backend(n, width, angles, cor)
    actual = backend._backproject_impl(y, angles)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(actual[backend.mask[None, ...] == 0], 0)


@pytest.mark.parametrize("cor", [-1.25, 0.375])
def test_cuda_oblique_orientation_and_cor_against_astra(gpu, cor):
    from tomobar.projectors.operator import ProjectionOperator
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry

    n = 64
    angles = np.array([-0.7, 0.3, 1.1, 2.4], dtype=np.float32)
    yy, xx = np.mgrid[:n, :n] - (n - 1) / 2
    phantom = np.exp(-((xx - 5) ** 2 + (yy + 4) ** 2) / 20)
    volume = gpu.asarray(phantom[None], dtype=gpu.float32)
    scan = Parallel3DGeometry(
        angles=angles, detector_shape=(1, n), centre_of_rotation=cor
    )
    grid = Volume3DGeometry((1, n, n))
    with ProjectionOperator(
        projection_geometry=scan, volume_geometry=grid, backend="astra"
    ) as astra:
        reference = astra.forward(volume)
    with ProjectionOperator(
        projection_geometry=scan, volume_geometry=grid, backend="usfft"
    ) as usfft:
        actual = usfft.forward(volume)
    # Smooth profiles suppress discretisation differences; their centroids
    # expose angle-sign, axis or centre-offset errors at oblique views.
    detector = gpu.arange(n, dtype=gpu.float32)
    expected_centroid = 5 * np.cos(angles) - 4 * np.sin(angles) + (n - 1) / 2 - cor
    centroid = lambda p: (p * detector).sum(axis=-1) / p.sum(axis=-1)
    np.testing.assert_allclose(
        gpu.asnumpy(centroid(actual))[0], expected_centroid, atol=2e-3
    )
    gpu.testing.assert_allclose(centroid(actual), centroid(reference), atol=0.05)
    assert (
        float(gpu.linalg.norm(actual - reference) / gpu.linalg.norm(reference)) < 0.03
    )
