"""Validate compatibility diagnostics against known matched/unmatched matrices."""

import numpy as np
import pytest

from tests.helpers.cg_diagnostics import diagnostics


def test_axis_aligned_source_model_is_matched():
    forward, backward = diagnostics.source_model_matrices([0, np.pi / 2])
    np.testing.assert_allclose(backward, forward.T, atol=1e-14)


def test_oblique_source_model_has_genuine_nonsymmetry():
    forward, backward = diagnostics.source_model_matrices(
        np.linspace(0, np.pi, 16, endpoint=False) + 0.03
    )
    metrics = diagnostics.matrix_metrics(
        forward, backward, np.eye(forward.shape[0]), 0.9
    )
    assert metrics["adjoint_relative_frobenius"] > 0.1
    assert metrics["adjoint_error_after_scalar_rescaling"] > 0.1
    assert metrics["cg_system_asymmetry"] > 0.04
    # Every p.T*K*p is positive, despite K being unsuitable for standard CG.
    assert metrics["minimum_symmetric_part_eigenvalue"] > 0
    target = diagnostics.quadratic_target_comparison(
        forward, backward, np.eye(forward.shape[0]), 0.9
    )
    assert target["relative_data_step_bias"] > 0.01
    assert target["mismatched_equation_stationarity"] < 1e-10
    assert target["intended_objective_stationarity"] > 0.01


def test_matched_control_converges_to_direct_solve():
    rng = np.random.default_rng(7)
    forward = rng.normal(size=(15, 6))
    trials = diagnostics.production_cg_trials(forward, forward.T, np.eye(15), 0.9)
    assert trials[-1]["true_relative_residual"] < 1e-9
    assert trials[-1]["relative_error_to_direct_system_solution"] < 1e-9


@pytest.mark.parametrize("method", ["LS", "PWLS", "SWLS"])
def test_precision_matrix_matches_full_fidelity(method):
    from tomobar.energy import data_energy
    from tomobar.options import DataFidelity

    rng = np.random.default_rng(9)
    weights = rng.uniform(0.1, 1, (2, 5, 3))
    residual = rng.normal(size=weights.shape)
    precision = diagnostics.precision_matrix(weights, method, beta=0.1)
    reference = data_energy(
        residual,
        np.zeros_like(residual),
        weights=None if method == "LS" else weights,
        fidelity=DataFidelity(method, beta_swls=0.1),
    )
    np.testing.assert_allclose(
        0.5 * residual.ravel() @ precision @ residual.ravel(), reference
    )
    np.testing.assert_allclose(precision, precision.T)
    assert np.linalg.eigvalsh(precision).min() >= -1e-12


def test_native_matrix_sampling_uses_working_operators(monkeypatch):
    import sys
    from types import SimpleNamespace

    instances = []

    class Reconstruction:
        def __init__(self, *, projection_geometry, volume_geometry, detector_padding):
            self.projection_shape = (
                2,
                len(projection_geometry.angles),
                12 + 2 * detector_padding,
            )
            rng = np.random.default_rng(3)
            self.forward_matrix = rng.normal(
                size=(np.prod(self.projection_shape), np.prod(volume_geometry.shape))
            ).astype(np.float32)
            self.backward_matrix = (self.forward_matrix.T + 0.01).copy()
            self.volume_shape = volume_geometry.shape
            self.operator = SimpleNamespace(
                forward=self._forward, backproject=self._backproject
            )
            instances.append(self)

        def _forward(self, image):
            return (self.forward_matrix @ image.ravel()).reshape(self.projection_shape)

        def _backproject(self, data):
            return (self.backward_matrix @ data.ravel()).reshape(self.volume_shape)

        def forward(self, *a, **kw):
            raise AssertionError("public crop must not enter compatibility checks")

        def backproject(self, *a, **kw):
            raise AssertionError(
                "public edge-padding must not enter compatibility checks"
            )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(zeros=np.zeros, float32=np.float32, asnumpy=np.asarray),
    )
    monkeypatch.setitem(
        sys.modules,
        "tomobar.reconstruction.iterative",
        SimpleNamespace(RecToolsIR=Reconstruction),
    )
    forward, backward, shape = diagnostics.dense_cuda_pair(
        np.array([0, 0.7]), padding=2
    )
    assert shape == (2, 2, 16)
    np.testing.assert_array_equal(forward, instances[0].forward_matrix)
    np.testing.assert_array_equal(backward, instances[0].backward_matrix)
