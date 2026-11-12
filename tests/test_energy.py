"""CPU objective checks and solver wiring without requiring ASTRA/CUDA."""

import ast
from pathlib import Path
from types import SimpleNamespace

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
def test_solver_history(name, enabled):
    # Execute the actual solver against an identity projector, replacing only
    # GPU infrastructure. This checks loop placement and all return paths.
    tree = ast.parse((Path(__file__).parents[1] / "tomobar/methodsIR.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    method.decorator_list = []
    method.returns = None
    for arg in method.args.args + method.args.kwonlyargs:
        arg.annotation = None
    from tomobar.supp.subsets import ProjectionSubsets

    def gradient(self, x, b, *args, **kwargs):
        return x - b

    ns = dict(
        np=np,
        cp=np,
        DataFidelity=DataFidelity,
        EnergyHistory=EnergyHistory,
        ProjectionSubsets=ProjectionSubsets,
        grad_data_term=gradient,
        _real=lambda v, *a: v,
        _boolean=lambda *a: None,
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<solver>", "exec"), ns)
    shape = (1, 2, 2)
    tool = SimpleNamespace(
        forward=lambda x, *a, **k: x.copy(), backproject=lambda x, *a, **k: x.copy()
    )
    recon = SimpleNamespace(
        _run_options=lambda iterations, **k: iterations,
        _prepare_inputs=lambda p, **k: (p, None),
        _initial_volume=lambda initial, ones=False: np.full(
            shape, int(ones), dtype=float
        ),
        _regularisation_options=lambda r: None,
        _finish_reconstruction=lambda x, **k: x,
        _Ax=tool.forward,
        _Atb=tool.backproject,
        Atools=tool,
        volume_geometry=SimpleNamespace(shape=shape),
        padded_projection_geometry=SimpleNamespace(projection_shape=shape),
        OS_number=1,
    )
    kwargs = dict(iterations=2, return_energy=enabled)
    if name in ("FISTA", "ADMM"):
        kwargs["lipschitz_constant"] = 2
    result = ns[name](recon, np.full(shape, 2.0), **kwargs)
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
