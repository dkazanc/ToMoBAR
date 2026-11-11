"""Typed solver contracts, warm starts, and immutable configuration reuse."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from tomobar.options import DataFidelity, Regularisation
from tests.test_reconstruction_geometry import harness
from tests.test_direct_filters import gpu


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
def test_invalid_solver_options_create_no_projector(harness, method, kwargs, error):
    backend, _, iterative = harness
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


def test_cgls_warm_start_uses_its_residual_and_preserves_storage(harness, monkeypatch):
    _, _, iterative = harness
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry

    scan = Parallel3DGeometry(angles=[0.0, 0.5, 1.0], detector_shape=(1, 5))
    grid = Volume3DGeometry(shape=(1, 2, 2))
    matrix = np.random.default_rng(0).random((15, 4)).astype(np.float32)
    projector = SimpleNamespace(
        close=Mock(),
        forward=lambda x, **kw: (matrix @ x.ravel()).reshape(scan.projection_shape),
        backproject=lambda b, **kw: (matrix.T @ b.ravel()).reshape(grid.shape),
    )
    monkeypatch.setattr(iterative, "AstraTools3D", lambda **kw: projector)
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
    from tomobar.methodsIR import RecToolsIR

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
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.methodsIR import RecToolsIR
    import tomobar.methodsIR as iterative

    seen = []

    def prox(rec, x, settings, **kwargs):
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
