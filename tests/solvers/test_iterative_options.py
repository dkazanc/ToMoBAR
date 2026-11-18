"""Typed solver contracts, warm starts, and immutable configuration reuse."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from tomobar.options import DataFidelity, Regularisation


@pytest.mark.parametrize(
    "factory,kwargs,error",
    [
        (DataFidelity, {"method": "unknown"}, ValueError),
        (DataFidelity, {"beta_swls": -1}, ValueError),
        (DataFidelity, {"beta_swls": np.nan}, ValueError),
        (Regularisation, {"method": "PD_TV_WAVELETS"}, ValueError),
        (Regularisation, {"strength": -1}, ValueError),
        (Regularisation, {"iterations": 0}, ValueError),
        (Regularisation, {"iterations": 1.5}, TypeError),
        (Regularisation, {"time_step": np.inf}, ValueError),
        (Regularisation, {"tv_type": 0.0}, ValueError),
        (Regularisation, {"half_precision": "yes"}, TypeError),
    ],
)
def test_settings_reject_invalid_values(factory, kwargs, error):
    with pytest.raises(error):
        factory(**kwargs)


def test_settings_are_immutable_and_reject_unknown_fields():
    for settings, field in ((DataFidelity(), "method"), (Regularisation(), "strength")):
        with pytest.raises(FrozenInstanceError):
            setattr(settings, field, None)
    with pytest.raises(TypeError):
        # Unsupported arguments must be rejected.
        Regularisation(regul_param=1.0)


@pytest.mark.parametrize(
    "method,kwargs,error",
    [
        ("CGLS", {"iterations": 0}, ValueError),
        ("CGLS", {"iterations": True}, TypeError),
        ("SIRT", {"nonnegativity": 1}, TypeError),
        ("Landweber", {"step_size": 0}, ValueError),
        ("FISTA", {"lipschitz_constant": 0}, ValueError),
        ("FISTA", {"fidelity": "PWLS"}, TypeError),
        ("FISTA", {"fidelity": DataFidelity("PWLS")}, ValueError),
        ("FISTA", {"regularisation": {"method": "PD_TV"}}, TypeError),
        ("ADMM", {"rho": -1}, ValueError),
        ("ADMM", {"relaxation": 2}, ValueError),
        ("CGLS", {"initial": np.zeros((1, 2, 2))}, ValueError),
        ("CGLS", {"iteration": 1}, TypeError),
        ("ADMM", {"tolerance": 0.1}, TypeError),
    ],
)
def test_invalid_solver_options_create_no_projector(
    mock_reconstruction, method, kwargs, error
):
    iterative = mock_reconstruction.iterative
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry

    scan = Parallel3DGeometry(angles=[0.0, 0.5, 1.0], detector_shape=(2, 7))
    with iterative.RecToolsIR(
        projection_geometry=scan, volume_geometry=Volume3DGeometry(shape=(2, 4, 6))
    ) as rec:
        with pytest.raises(error):
            getattr(rec, method)(
                np.ones(scan.projection_shape, dtype=np.float32), **kwargs
            )
        assert not rec.Atools._projector_handles


def test_cgls_warm_start_uses_its_residual_and_preserves_storage(
    mock_reconstruction, monkeypatch
):
    iterative = mock_reconstruction.iterative
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry

    scan = Parallel3DGeometry(angles=[0.0, 0.5, 1.0], detector_shape=(1, 5))
    grid = Volume3DGeometry(shape=(1, 2, 2))
    matrix = np.random.default_rng(0).random((15, 4)).astype(np.float32)
    projector = SimpleNamespace(
        close=Mock(),
        forward=lambda x, **kw: (matrix @ x.ravel()).reshape(scan.projection_shape),
        backproject=lambda b, **kw: (matrix.T @ b.ravel()).reshape(grid.shape),
    )
    monkeypatch.setattr(iterative, "create_projection_backend", lambda **kw: projector)
    initial = np.arange(4, dtype=np.float32).reshape(grid.shape) / 10
    before = initial.copy()
    data = np.ones(scan.projection_shape, dtype=np.float32)
    residual = data.ravel() - matrix @ initial.ravel()
    direction = matrix.T @ residual
    projected = matrix @ direction
    expected = (
        initial.ravel() + (direction @ direction) / (projected @ projected) * direction
    )
    with iterative.RecToolsIR(projection_geometry=scan, volume_geometry=grid) as rec:
        result = rec.CGLS(data, initial=initial, iterations=1, recon_mask_radius=None)
        np.testing.assert_allclose(result.ravel(), expected, rtol=2e-6)
        np.testing.assert_array_equal(initial, before)
        exact_data = projector.forward(initial)
        np.testing.assert_array_equal(
            rec.CGLS(exact_data, initial=initial, iterations=1, recon_mask_radius=None),
            initial,
        )


@pytest.mark.parametrize(
    "method", ["Landweber", "SIRT", "CGLS", "FISTA", "ADMM", "OSEM"]
)
def test_gpu_warm_starts_are_preserved(gpu, method):
    cp = gpu
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.reconstruction.iterative import RecToolsIR

    scan = Parallel3DGeometry(angles=np.linspace(0, np.pi, 7), detector_shape=(3, 9))
    grid = Volume3DGeometry(shape=(3, 6, 8))
    initial = cp.asarray(
        np.random.default_rng(0).random(grid.shape).astype(np.float32) / 10
    )
    original = initial.copy()
    data = cp.ones(scan.projection_shape, dtype=cp.float32)
    options = {"lipschitz_constant": 1000.0} if method in ("FISTA", "ADMM") else {}
    with RecToolsIR(projection_geometry=scan, volume_geometry=grid) as rec:
        result = getattr(rec, method)(
            data, iterations=1, initial=initial, recon_mask_radius=None, **options
        )
        assert result.shape == grid.shape
        assert cp.isfinite(result).all()
        cp.testing.assert_array_equal(initial, original)
        cp.testing.assert_array_equal(data, cp.ones_like(data))


def test_admm_scales_a_local_regularisation_copy(gpu, monkeypatch):
    cp = gpu
    import tomobar.reconstruction.iterative as iterative
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.reconstruction.iterative import RecToolsIR

    seen = []

    def prox(x, settings, **kwargs):
        seen.append(settings)
        return x

    monkeypatch.setattr(iterative, "prox_regul", prox)
    scan = Parallel3DGeometry(angles=np.linspace(0, np.pi, 7), detector_shape=(3, 9))
    settings = Regularisation(strength=0.04)
    with RecToolsIR(
        projection_geometry=scan, volume_geometry=Volume3DGeometry(shape=(3, 6, 8))
    ) as rec:
        data = cp.ones(scan.projection_shape, dtype=cp.float32)
        first = rec.ADMM(
            data, iterations=1, lipschitz_constant=1000, rho=2, regularisation=settings
        )
        second = rec.ADMM(
            data, iterations=1, lipschitz_constant=1000, rho=2, regularisation=settings
        )
        cp.testing.assert_array_equal(first, second)
    assert settings.strength == 0.04
    assert len(seen) == 2
    assert all(option.strength == 0.02 and option is not settings for option in seen)


@pytest.mark.parametrize("fidelity", ["PWLS", "SWLS"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, -1.0, 0.0])
def test_weight_validation_precedes_projection(mock_reconstruction, fidelity, bad):
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry

    scan = Parallel3DGeometry(angles=[0.0, 0.5, 1.0], detector_shape=(1, 3))
    with mock_reconstruction.iterative.RecToolsIR(
        projection_geometry=scan, volume_geometry=Volume3DGeometry(shape=(1, 2, 2))
    ) as rec:
        weights = np.full(scan.projection_shape, bad, dtype=np.float32)
        with pytest.raises(ValueError, match="weights"):
            rec.FISTA(
                np.ones_like(weights), weights=weights, fidelity=DataFidelity(fidelity)
            )
        assert not rec.Atools._projector_handles
        np.testing.assert_array_equal(weights, np.full_like(weights, bad))


@pytest.mark.parametrize("fidelity", ["PWLS", "SWLS"])
def test_weights_preserve_zeros_and_relative_values(mock_reconstruction, fidelity):
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry

    scan = Parallel3DGeometry(angles=[0.0, 0.5, 1.0], detector_shape=(1, 3))
    weights = np.array([[[0, 2, 4], [1e-9, 3, 1], [2, 0, 4]]], dtype=np.float32)
    original = weights.copy()
    with mock_reconstruction.iterative.RecToolsIR(
        projection_geometry=scan,
        volume_geometry=Volume3DGeometry(shape=(1, 2, 2)),
        detector_padding=1,
    ) as rec:
        _, prepared = rec._prepare_inputs(
            np.ones_like(weights), weights=weights, fidelity=DataFidelity(fidelity)
        )
        expected = np.pad(original / 4, ((0, 0), (0, 0), (1, 1)), mode="edge")
        np.testing.assert_array_equal(prepared, expected)
        assert not np.shares_memory(prepared, weights)
    np.testing.assert_array_equal(weights, original)


@pytest.mark.parametrize("subset_count", [1, 2, 3])
def test_power_method_covers_every_subset(
    mock_reconstruction, monkeypatch, subset_count
):
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry

    iterative = mock_reconstruction.iterative
    scan = Parallel3DGeometry(angles=np.arange(5), detector_shape=(1, 2))
    grid = Volume3DGeometry(shape=(1, 1, 2))
    # Different diagonal blocks ensure the first subset is not the largest.
    blocks = [np.diag([1.0, scale]) for scale in (2.0, 9.0, 4.0)[:subset_count]]
    seen = set()

    def block(subset):
        return np.vstack(blocks) if subset is None else blocks[subset]

    def forward(x, *, subset_index=None):
        seen.add(subset_index)
        return block(subset_index) @ x.ravel()

    projector = SimpleNamespace(
        forward=forward,
        backproject=lambda b, subset_index=None: (block(subset_index).T @ b).reshape(
            grid.shape
        ),
        close=Mock(),
    )
    monkeypatch.setattr(iterative, "create_projection_backend", lambda **kw: projector)
    with iterative.RecToolsIR(
        projection_geometry=scan, volume_geometry=grid, num_subsets=subset_count
    ) as rec:
        estimate = rec.powermethod(iterations=30)
    expected = max(np.linalg.norm(A, 2) ** 2 for A in blocks)
    np.testing.assert_allclose(estimate, expected, rtol=1e-6)
    assert seen == (set(range(subset_count)) if subset_count > 1 else {None})


def test_power_method_skips_zero_subset(mock_reconstruction, monkeypatch):
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry

    iterative = mock_reconstruction.iterative
    scan = Parallel3DGeometry(angles=[0.0, 1.0], detector_shape=(1, 2))
    grid = Volume3DGeometry(shape=(1, 1, 2))
    projector = SimpleNamespace(
        forward=lambda x, subset_index=None: x * (0 if subset_index == 0 else 3),
        backproject=lambda b, subset_index=None: b * (0 if subset_index == 0 else 3),
        close=Mock(),
    )
    monkeypatch.setattr(iterative, "create_projection_backend", lambda **kw: projector)
    with iterative.RecToolsIR(
        projection_geometry=scan, volume_geometry=grid, num_subsets=2
    ) as rec:
        np.testing.assert_allclose(rec.powermethod(), 9.0, rtol=1e-6)
        projector.backproject = lambda b, **kw: np.zeros(grid.shape, dtype=np.float32)
        with pytest.raises(ValueError, match="positive"):
            rec.powermethod()
        projector.backproject = lambda b, **kw: np.full(grid.shape, np.nan)
        with pytest.raises(ValueError, match="finite"):
            rec.powermethod()


def test_admm_power_method_uses_full_operator(mock_reconstruction, monkeypatch):
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry

    iterative = mock_reconstruction.iterative
    scan = Parallel3DGeometry(angles=np.arange(5), detector_shape=(1, 2))
    grid = Volume3DGeometry(shape=(1, 1, 2))
    matrix = np.diag([1.0, 9.0])
    seen = []

    def forward(image, *, subset_index=None):
        seen.append(subset_index)
        return matrix @ image.ravel()

    projector = SimpleNamespace(
        forward=forward,
        backproject=lambda data, **kw: (matrix.T @ data).reshape(grid.shape),
        close=Mock(),
    )
    monkeypatch.setattr(iterative, "create_projection_backend", lambda **kw: projector)
    with iterative.RecToolsIR(
        projection_geometry=scan, volume_geometry=grid, num_subsets=3
    ) as rec:
        state = rec.initialise_solver(
            "ADMM", np.ones(scan.projection_shape, dtype=np.float32)
        )
        np.testing.assert_allclose(
            state.algorithm_state["lipschitz_constant"], 81, rtol=1e-6
        )
        assert set(seen) == {None}
        with pytest.raises(TypeError, match="full_data"):
            rec.powermethod(full_data=1)
