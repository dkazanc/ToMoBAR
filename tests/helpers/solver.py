"""CPU reconstruction factory and exact proximal reference for solver tests."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tomobar._solvers import state as solver_state
from tomobar.reconstruction.state import SolverState
from tomobar.options import DataFidelity, Diagnostics, Regularisation
from tomobar._validation import _positive_integer, _real


def make_cpu_rec():
    # Load public methods without CUDA decorators; numerical steps are the
    # production state engine using NumPy in place of CuPy.
    tree = ast.parse(
        (Path(__file__).parents[2] / "tomobar/reconstruction/iterative.py").read_text()
    )
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    names = {
        "initialise_solver",
        "_check_solver_backend",
        "run",
        "solver_image",
        "_solve",
        "Landweber",
        "SIRT",
        "CGLS",
        "FISTA",
        "ADMM",
        "OSEM",
    }
    cls.body = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    cls.bases = []
    for node in cls.body:
        node.decorator_list = []
        node.returns = None
        for arg in node.args.args + node.args.kwonlyargs:
            arg.annotation = None

    def gradient(
        rec,
        x,
        b,
        use_os,
        sub,
        weights,
        weight_sum,
        *,
        fidelity,
        projection_out=None,
        out=None,
    ):
        r = rec.operator.forward(
            x, subset_index=sub if use_os else None, out=projection_out
        )
        if b is not None:
            r -= b
        if weights is not None:
            r = weights * r
            if fidelity == "SWLS":
                r -= weights * (r.sum(axis=1) / weight_sum)[:, None, :]
        return rec.operator.backproject(
            r, subset_index=sub if use_os else None, out=out
        )

    ns = dict(
        np=np,
        cp=np,
        DataFidelity=DataFidelity,
        Diagnostics=Diagnostics,
        solver_state=solver_state,
        SolverState=SolverState,
        _real=_real,
        grad_data_term=gradient,
        prox_regul=lambda x, reg, **kw: x / (1 + reg.strength),
    )
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "<cpu_rec>", "exec"), ns)

    class CPURec(ns["RecToolsIR"]):
        backend = "astra"
        device_index = 0
        OS_number = 1
        geometry_version = 0
        closed = False
        scale = 1.0
        preparations = 0
        power_calls = 0
        projection_calls = 0
        shape = (1, 3, 2)
        volume_geometry = SimpleNamespace(shape=shape)

        def __init__(self):
            self.operator = SimpleNamespace(
                forward=self._forward, backproject=self._backproject
            )

        def _ensure_open(self):
            if self.closed:
                raise RuntimeError("closed")

        def _run_options(self, iterations, *, default, allow_subsets=True, **kw):
            self._ensure_open()
            if not allow_subsets and self.OS_number > 1:
                raise ValueError("subsets")
            return _positive_integer(
                default if iterations is None else iterations, "iterations"
            )

        def _regularisation_options(self, r):
            if r is not None and not isinstance(r, Regularisation):
                raise TypeError("regularisation")

        def _prepare_inputs(self, p, weights=None, **kw):
            self.preparations += 1
            return p, weights

        def _initial_volume(self, initial, *, ones=False):
            return (
                np.full(self.shape, int(ones), dtype=np.float32)
                if initial is None
                else initial.copy()
            )

        def _forward(self, x, *, out=None, subset_index=None):
            self.projection_calls += 1
            y = self.scale * np.arange(1, 7, dtype=np.float32).reshape(self.shape) * x
            result = (
                y[:, subset_index :: self.OS_number, :]
                if subset_index is not None
                else y
            )
            if out is not None:
                out[...] = result
                return out
            return result

        def _backproject(self, b, *, out=None, subset_index=None):
            x = np.zeros(self.shape, dtype=np.float32)
            if subset_index is not None:
                x[:, subset_index :: self.OS_number, :] = b
            else:
                x[:] = b
            result = (
                self.scale * np.arange(1, 7, dtype=np.float32).reshape(self.shape) * x
            )
            if out is not None:
                out[...] = result
                return out
            return result

        def powermethod(self, *, full_data=False):
            self.power_calls += 1
            return 36 * self.scale**2

        def _finish_reconstruction(self, x, recon_mask_radius=None):
            if recon_mask_radius is not None:
                x[:, :, 0] = 0
            return x[:, :2, :]

    return CPURec()


def two_pixel_tv_prox(image, regularisation, **kwargs):
    """Exact prox of strength * abs(x[1] - x[0]), independent of GPU TV code."""
    left, right = image.ravel()
    gap = float(right - left)
    correction = np.sign(gap) * min(regularisation.strength, abs(gap) / 2)
    result = np.array(
        [left + correction, right - correction], dtype=image.dtype
    ).reshape(image.shape)
    if kwargs.get("nonnegativity", False):
        np.maximum(result, 0, out=result)
    return result
