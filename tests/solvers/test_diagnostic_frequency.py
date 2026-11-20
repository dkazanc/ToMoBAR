"""Sparse diagnostics preserve solver updates and continuation schedules."""

from dataclasses import FrozenInstanceError
from unittest.mock import Mock

import numpy as np
import pytest

from tomobar.options import Diagnostics


@pytest.mark.parametrize(
    "field", ["energy_interval", "residual_interval", "convergence_interval"]
)
@pytest.mark.parametrize(
    "value,error",
    [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)],
)
def test_intervals_require_positive_integers(field, value, error):
    with pytest.raises(error, match=field):
        Diagnostics(**{field: value})


def test_settings_are_immutable_and_only_reporting_can_be_disabled():
    settings = Diagnostics(residual_interval=None)
    with pytest.raises(FrozenInstanceError):
        settings.energy_interval = 2
    for field in ("energy_interval", "convergence_interval"):
        with pytest.raises(TypeError, match=field):
            Diagnostics(**{field: None})


@pytest.mark.parametrize(
    "method", ["Landweber", "SIRT", "CGLS", "FISTA", "ADMM", "OSEM"]
)
def test_energy_schedule_is_independent_of_run_boundaries(cpu_rec, method):
    rec = cpu_rec
    data = np.arange(2, 8, dtype=np.float32).reshape(rec.shape)
    options = dict(return_energy=True, diagnostics=Diagnostics(energy_interval=3))
    whole = rec.initialise_solver(method, data, **options)
    split = rec.initialise_solver(method, data, **options)
    rec.run(whole, iterations=9)
    rec.run(split, iterations=2)
    rec.run(split, iterations=7)
    assert whole.energy_iterations == list(range(0, whole.iteration + 1, 3))
    assert split.energy_iterations == whole.energy_iterations
    np.testing.assert_array_equal(split.energy, whole.energy)
    np.testing.assert_array_equal(split.image, whole.image)


def test_energy_records_geometry_restart_even_between_scheduled_samples(cpu_rec):
    rec = cpu_rec
    state = rec.initialise_solver(
        "FISTA",
        np.ones(rec.shape, dtype=np.float32),
        return_energy=True,
        diagnostics=Diagnostics(energy_interval=3),
    )
    rec.run(state, iterations=2)
    rec.geometry_version += 1
    rec.scale = 1.2
    rec.run(state)
    assert state.energy_iterations == [0, 2, 3]
    assert state.energy_versions == [0, 1, 1]
    assert len(state.energy_times) == 3


def test_disabled_reporting_skips_norms_and_preserves_admm_updates(
    cpu_rec, monkeypatch
):
    rec = cpu_rec
    data = np.arange(2, 8, dtype=np.float32).reshape(rec.shape)
    expected = rec.initialise_solver("ADMM", data)
    rec.run(expected, iterations=10)
    actual = rec.initialise_solver(
        "ADMM", data, diagnostics=Diagnostics(residual_interval=None)
    )
    norm = Mock(side_effect=AssertionError("unscheduled norm evaluation"))
    monkeypatch.setattr(np.linalg, "norm", norm)
    rec.run(actual, iterations=3)
    rec.run(actual, iterations=7)
    norm.assert_not_called()
    assert actual.residuals["iteration"].size == 0
    np.testing.assert_array_equal(actual.image, expected.image)


def test_sparse_residuals_only_evaluate_norms_on_reported_iterations(
    cpu_rec, monkeypatch
):
    rec = cpu_rec
    state = rec.initialise_solver(
        "ADMM",
        np.ones(rec.shape, dtype=np.float32),
        diagnostics=Diagnostics(residual_interval=4),
    )
    norm = Mock(wraps=np.linalg.norm)
    monkeypatch.setattr(np.linalg, "norm", norm)
    rec.run(state, iterations=3)
    rec.run(state, iterations=7)
    np.testing.assert_array_equal(state.residuals["iteration"], [4, 8])
    assert norm.call_count == 6  # Three norms at each reporting event.


def test_convergence_checks_are_independent_of_reporting(cpu_rec, capsys):
    rec = cpu_rec
    state = rec.initialise_solver(
        "ADMM",
        np.ones(rec.shape, dtype=np.float32),
        absolute_tolerance=1e6,
        relative_tolerance=1e6,
        verbose=True,
        diagnostics=Diagnostics(residual_interval=None, convergence_interval=3),
    )
    rec.run(state, iterations=2)
    assert not state.converged
    rec.run(state, iterations=10)
    assert state.converged and state.iteration == 3
    assert state.residuals["iteration"].size == 0
    assert capsys.readouterr().out == ""


def test_invalid_settings_fail_before_preparing_data(cpu_rec):
    with pytest.raises(TypeError, match="Diagnostics"):
        cpu_rec.initialise_solver("FISTA", np.ones(cpu_rec.shape), diagnostics={})
    assert cpu_rec.preparations == 0


def test_gpu_disabled_reporting_skips_norms_and_preserves_images(gpu, monkeypatch):
    cp = gpu
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.reconstruction.iterative import IterativeReconstructor

    shape = (3, 8, 8)
    scan = Parallel3DGeometry(
        angles=np.linspace(0, np.pi, 7, endpoint=False), detector_shape=(3, 10)
    )
    with IterativeReconstructor(
        projection_geometry=scan, volume_geometry=Volume3DGeometry(shape)
    ) as rec:
        data = rec.forward(cp.ones(shape, dtype=cp.float32))
        expected = rec.initialise_solver("ADMM", data, lipschitz_constant=1000)
        rec.run(expected, iterations=5)
        actual = rec.initialise_solver(
            "ADMM",
            data,
            lipschitz_constant=1000,
            diagnostics=Diagnostics(residual_interval=None),
        )
        norm = Mock(side_effect=AssertionError("unscheduled GPU norm"))
        monkeypatch.setattr(cp.linalg, "norm", norm)
        rec.run(actual, iterations=5)
        norm.assert_not_called()
        cp.testing.assert_array_equal(actual.image, expected.image)
