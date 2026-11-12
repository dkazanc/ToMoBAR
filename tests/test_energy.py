"""CPU objective checks and solver wiring without requiring ASTRA/CUDA."""

from tests.test_solver_state import cpu_rec

import numpy as np
import pytest

from tomobar.energy import EnergyHistory, data_energy, regularisation_energy
from tomobar.options import DataFidelity, Regularisation


@pytest.mark.parametrize("method", ["LS", "PWLS", "SWLS", "KL"])
def test_fidelity_gradient(method):
    rng = np.random.default_rng(12)
    x = rng.uniform(1, 2, (2, 3, 4))
    b = rng.uniform(1, 2, x.shape)
    w = rng.uniform(0.1, 1, x.shape) if method in ("PWLS", "SWLS") else None
    fidelity = DataFidelity(method=method)
    r = x - b
    gradient = r if w is None else w * r
    if method == "KL":
        gradient = 1 - b / x
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


def test_tv_and_zero_data():
    x = np.array([[0.0, 1.0], [1.0, 2.0]])
    assert regularisation_energy(x, Regularisation(strength=2, tv_type=1)) == 8
    np.testing.assert_allclose(
        regularisation_energy(x, Regularisation(strength=2)), 2 * (2 + np.sqrt(2))
    )
    assert (
        data_energy(
            np.ones((1, 2, 1)), np.zeros((1, 2, 1)), fidelity=DataFidelity(method="KL")
        )
        == 2
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
    recon._Ax = lambda x, *a, **k: x.copy()
    recon._Atb = lambda x, *a, **k: x.copy()
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
        np.testing.assert_allclose(
            energy[-1],
            data_energy(
                image,
                np.full(shape, 2.0),
                fidelity=DataFidelity(method="KL" if name == "OSEM" else "LS"),
            ),
        )
    else:
        assert result.shape == shape
