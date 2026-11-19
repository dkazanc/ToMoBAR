"""CPU objective checks and solver wiring without requiring ASTRA/CUDA."""

import numpy as np
import pytest

from tomobar.energy import EnergyHistory, data_energy, regularisation_energy
from tomobar.options import DataFidelity, Regularisation


@pytest.mark.essential
@pytest.mark.parametrize("method", ["LS", "PWLS", "SWLS"])
def test_fidelity_gradient(method):
    rng = np.random.default_rng(12)
    x = rng.uniform(1, 2, (2, 3, 4))
    b = rng.uniform(1, 2, x.shape)
    w = rng.uniform(0.1, 1, x.shape) if method in ("PWLS", "SWLS") else None
    fidelity = DataFidelity(method=method)
    r = x - b
    gradient = r if w is None else w * r
    if method == "SWLS":
        gradient -= (
            w
            * (np.sum(w * r, axis=1) / (np.sum(w, axis=1) + fidelity.beta_swls))[
                :, None, :
            ]
        )
    direction = rng.normal(size=x.shape)

    def cost(v):
        return data_energy(v, b, fidelity=fidelity, weights=w)

    numerical = (cost(x + 1e-5 * direction) - cost(x - 1e-5 * direction)) / 2e-5
    np.testing.assert_allclose(numerical, np.sum(gradient * direction), rtol=1e-7)


@pytest.mark.essential
def test_tv_energy():
    x = np.array([[0.0, 1.0], [1.0, 2.0]])
    assert regularisation_energy(x, Regularisation(strength=2, tv_type=1)) == 8
    np.testing.assert_allclose(
        regularisation_energy(x, Regularisation(strength=2)), 2 * (2 + np.sqrt(2))
    )


def test_disabled_history_never_projects():
    h = EnergyHistory(False, None, None, array_module=np)
    h.record(None)
    image = np.ones(2)
    assert h.finish(image) is image


@pytest.mark.parametrize("name", ["Landweber", "SIRT", "CGLS", "FISTA", "ADMM", "OSEM"])
@pytest.mark.parametrize("enabled", [False, True])
def test_solver_history(name, enabled, cpu_rec):
    recon = cpu_rec
    # Use an identity operator to exercise the exact CGLS early-exit path.
    recon.operator.forward = lambda x, *a, **k: x.copy()
    recon.operator.backproject = lambda x, *a, **k: x.copy()
    recon._finish_reconstruction = lambda x, **k: x
    shape = recon.shape
    kwargs = dict(iterations=2, return_energy=enabled)
    if name in ("FISTA", "ADMM"):
        kwargs["lipschitz_constant"] = 2
    result = getattr(recon, name)(np.full(shape, 2.0), recon_mask_radius=None, **kwargs)
    if enabled:
        image, energy = result
        assert energy.ndim == 1
        assert len(energy) == (2 if name == "CGLS" else 3)
        if name == "OSEM":
            # OSEM reports its own emission likelihood, not a selectable fidelity.
            expected = np.sum(image - 2.0 * np.log(np.maximum(image, 1e-8)))
        else:
            expected = data_energy(image, np.full(shape, 2.0))
        np.testing.assert_allclose(energy[-1], expected)
    else:
        assert result.shape == shape


def test_energy_times_use_completed_samples_and_disabled_history_has_no_clock(
    monkeypatch,
):
    from types import SimpleNamespace

    import tomobar.energy as energy_module

    clock = iter([20.0, 20.25, 22.0])
    monkeypatch.setattr(energy_module, "perf_counter", lambda: next(clock))
    recorder = EnergyHistory(
        True,
        SimpleNamespace(operator=SimpleNamespace(forward=lambda x: x)),
        np.zeros(2),
        array_module=np,
    )
    for value in (3, 2, 1):
        recorder.record(np.full(2, value))
    np.testing.assert_array_equal(recorder.times, [0, 0.25, 2])
    np.testing.assert_array_equal(recorder.finish(None)[1], [9, 4, 1])
    # The clock iterator is now exhausted; disabled recording must not read it.
    disabled = EnergyHistory(False, None, None, array_module=np)
    disabled.record(None)
    assert disabled.times == []


@pytest.mark.parametrize(
    "method", ["Landweber", "SIRT", "CGLS", "FISTA", "ADMM", "OSEM"]
)
def test_solver_energy_times_follow_history_and_geometry(cpu_rec, monkeypatch, method):
    from itertools import count

    import tomobar.energy as energy_module

    clock = count(step=0.5)
    monkeypatch.setattr(energy_module, "perf_counter", lambda: next(clock))
    rec = cpu_rec
    options = {"lipschitz_constant": 40} if method in ("FISTA", "ADMM") else {}
    data = np.ones(rec.shape, dtype=np.float32)
    state = rec.initialise_solver(method, data, return_energy=True, **options)
    rec.run(state, iterations=2)
    rec.geometry_version += 1
    rec.run(state)
    np.testing.assert_array_equal(
        state.energy_times, np.arange(len(state.energy)) * 0.5
    )
    assert (
        len(state.energy_times)
        == len(state.energy_versions)
        == len(state.energy_iterations)
    )
    times = state.energy_times
    times[:] = -1
    assert np.all(state.energy_times >= 0)
    disabled = rec.initialise_solver(method, data, **options)
    rec.run(disabled)
    assert disabled.energy_times is None


def test_timestamp_is_taken_after_energy_scalar_is_ready(monkeypatch):
    from types import SimpleNamespace

    import tomobar.energy as energy_module

    completed = []

    class DeferredScalar:
        def __add__(self, other):
            return self

        def __float__(self):
            completed.append(True)
            return 1.0

    def clock():
        assert completed
        return 1.0

    monkeypatch.setattr(energy_module, "data_energy", lambda *a, **kw: DeferredScalar())
    monkeypatch.setattr(energy_module, "perf_counter", clock)
    recorder = EnergyHistory(
        True,
        SimpleNamespace(operator=SimpleNamespace(forward=lambda x: x)),
        np.zeros(2),
        array_module=np,
    )
    recorder.record(np.ones(2))
    np.testing.assert_array_equal(recorder.finish(None)[1], [1])
    np.testing.assert_array_equal(recorder.times, [0])
