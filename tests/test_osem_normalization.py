"""Independent emission-EM references: sensitivity direction and subset ownership."""

from unittest.mock import Mock
import weakref

import numpy as np
import pytest

from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry

SYSTEM_MATRIX = np.array(
    [[1, 2, 0.5], [3, 1, 2], [0.5, 4, 1], [5, 0.5, 3], [2, 3, 4]], dtype=np.float32
)
TRUE_VOXELS = np.array([1, 0.5, 2], dtype=np.float32)


def reference_osem(matrix, measured, initial, subsets, iterations):
    """Dense float64 EM, independent of production projection/subset helpers."""
    estimate = np.asarray(initial, dtype=np.float64).ravel().copy()
    matrix = matrix.astype(np.float64)
    measured = np.asarray(measured, dtype=np.float64).ravel()
    for _ in range(iterations):
        for index in range(subsets):
            operator = matrix[index::subsets]
            sensitivity = operator.sum(axis=0)
            ratio = measured[index::subsets] / np.maximum(operator @ estimate, 1e-8)
            estimate *= (operator.T @ ratio) / np.maximum(sensitivity, 1e-8)
    return estimate


class MatrixProjector:
    """Exact adjoint pair with tracked sensitivity lifetimes, without CUDA."""

    def __init__(self, matrix, subsets):
        self.matrix, self.subsets = matrix, subsets
        self.shape = (1, 1, matrix.shape[1])
        self.close = Mock()
        self.next_backprojection_is_data = False
        self.sensitivities = []
        self.previous_sensitivity_alive = []

    def selected(self, subset):
        return self.matrix if subset is None else self.matrix[subset :: self.subsets]

    def forward(self, volume, *, subset_index=None):
        self.next_backprojection_is_data = True
        return (self.selected(subset_index) @ volume.ravel()).reshape(1, -1, 1)

    def backproject(self, projections, *, subset_index=None):
        result = (self.selected(subset_index).T @ projections.ravel()).reshape(
            self.shape
        )
        if not self.next_backprojection_is_data:
            self.previous_sensitivity_alive.append(
                any(ref() is not None for ref in self.sensitivities)
            )
            self.sensitivities.append(weakref.ref(result))
        self.next_backprojection_is_data = False
        return result


@pytest.fixture
def matrix_reconstructor(mock_reconstruction, monkeypatch):
    """Create an exact matrix-backed reconstructor for each numerical example."""
    iterative = mock_reconstruction.iterative

    def make(matrix, subsets):
        projector = MatrixProjector(matrix, subsets)
        monkeypatch.setattr(iterative, "AstraTools3D", lambda **kwargs: projector)
        scan = Parallel3DGeometry(
            angles=np.linspace(0, np.pi, matrix.shape[0], endpoint=False),
            detector_shape=(1, 1),
        )
        rec = iterative.RecToolsIR(
            projection_geometry=scan,
            volume_geometry=Volume3DGeometry(shape=projector.shape),
            OS_number=subsets,
        )
        return rec, projector

    return make


@pytest.mark.parametrize(
    "subsets",
    [1, 2, 3, 5],
    ids=["mlem", "two-uneven-subsets", "three-uneven-subsets", "one-view-per-subset"],
)
@pytest.mark.parametrize("iterations", [1, 4], ids=["one-sweep", "four-sweeps"])
def test_osem_matches_independent_dense_reference(
    matrix_reconstructor, subsets, iterations
):
    measured = SYSTEM_MATRIX @ TRUE_VOXELS
    initial = np.array([0.8, 0.7, 1.2], dtype=np.float32)
    data, initial_volume = measured.reshape(1, -1, 1), initial.reshape(1, 1, -1)
    original_data, original_initial = data.copy(), initial_volume.copy()
    expected = reference_osem(SYSTEM_MATRIX, measured, initial, subsets, iterations)
    rec, projector = matrix_reconstructor(SYSTEM_MATRIX, subsets)
    with rec:
        result = rec.OSEM(
            data, initial=initial_volume, iterations=iterations, recon_mask_radius=None
        )
        np.testing.assert_allclose(result.ravel(), expected, rtol=3e-6, atol=2e-7)
        np.testing.assert_array_equal(data, original_data)
        np.testing.assert_array_equal(initial_volume, original_initial)
    projector.close.assert_called_once()


@pytest.mark.parametrize(
    "subsets",
    [1, 2, 3, 5],
    ids=["mlem", "two-uneven-subsets", "three-uneven-subsets", "one-view-per-subset"],
)
def test_osem_preserves_exact_noiseless_solution(matrix_reconstructor, subsets):
    rec, _ = matrix_reconstructor(SYSTEM_MATRIX, subsets)
    with rec:
        result = rec.OSEM(
            (SYSTEM_MATRIX @ TRUE_VOXELS).reshape(1, -1, 1),
            initial=TRUE_VOXELS.reshape(1, 1, -1),
            iterations=3,
            recon_mask_radius=None,
        )
    np.testing.assert_allclose(result.ravel(), TRUE_VOXELS, rtol=2e-6, atol=2e-7)


def test_mlem_two_voxel_regression(matrix_reconstructor):
    matrix = np.array([[1.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    rec, _ = matrix_reconstructor(matrix, 1)
    with rec:
        result = rec.OSEM(
            np.array([2.0, 3.0], dtype=np.float32).reshape(1, 2, 1),
            initial=np.ones((1, 1, 2), dtype=np.float32),
            iterations=1,
            recon_mask_radius=None,
        )
    # Multiplying by sensitivity previously returned [4, 9].
    np.testing.assert_array_equal(result.ravel(), [1.0, 1.0])


@pytest.mark.parametrize(
    "subsets",
    [1, 2, 3, 5],
    ids=["mlem", "two-uneven-subsets", "three-uneven-subsets", "one-view-per-subset"],
)
def test_osem_zero_sensitivity_is_finite_and_zero(matrix_reconstructor, subsets):
    matrix = np.column_stack((SYSTEM_MATRIX, np.zeros(5, dtype=np.float32)))
    truth = np.append(TRUE_VOXELS, np.float32(0))
    initial = np.array([0.8, 0.7, 1.2, 3], dtype=np.float32)
    measured = matrix @ truth
    expected = reference_osem(matrix, measured, initial, subsets, 3)
    rec, _ = matrix_reconstructor(matrix, subsets)
    with rec, np.errstate(divide="raise", invalid="raise"):
        result = rec.OSEM(
            measured.reshape(1, -1, 1),
            initial=initial.reshape(1, 1, -1),
            iterations=3,
            recon_mask_radius=None,
        )
    assert np.isfinite(result).all()
    assert result.ravel()[-1] == 0
    np.testing.assert_allclose(result.ravel(), expected, rtol=3e-6, atol=2e-7)


@pytest.mark.parametrize("subsets", [1, 3])
def test_sensitivity_storage_is_bounded_and_mlem_reuses_it(
    matrix_reconstructor, subsets
):
    rec, projector = matrix_reconstructor(SYSTEM_MATRIX, subsets)
    with rec:
        rec.OSEM(
            (SYSTEM_MATRIX @ TRUE_VOXELS).reshape(1, -1, 1),
            iterations=3,
            recon_mask_radius=None,
        )
    assert len(projector.sensitivities) == (1 if subsets == 1 else 3 * subsets)
    assert not any(projector.previous_sensitivity_alive)
    assert all(ref() is None for ref in projector.sensitivities)


def test_mlem_likelihood_improves(matrix_reconstructor):
    measured = SYSTEM_MATRIX @ TRUE_VOXELS
    initial = np.array([0.2, 1.5, 0.3], dtype=np.float32)
    losses = []
    rec, _ = matrix_reconstructor(SYSTEM_MATRIX, 1)
    with rec:
        for iterations in range(1, 6):
            result = rec.OSEM(
                measured.reshape(1, -1, 1),
                initial=initial.reshape(1, 1, -1),
                iterations=iterations,
                recon_mask_radius=None,
            )
            predicted = SYSTEM_MATRIX.astype(np.float64) @ result.ravel()
            losses.append(np.sum(predicted - measured * np.log(predicted)))
    assert np.all(np.diff(losses) <= 1e-7)


@pytest.mark.parametrize("subsets", [1, 3, 7])
def test_gpu_osem_preserves_exact_solution_and_input_arrays(gpu, subsets):
    cp = gpu
    from tomobar.methodsIR import RecToolsIR

    scan = Parallel3DGeometry(
        angles=np.linspace(0, np.pi, 7, endpoint=False), detector_shape=(3, 12)
    )
    grid = Volume3DGeometry(shape=(3, 6, 8))
    truth = cp.asarray(
        np.linspace(0.2, 1, np.prod(grid.shape), dtype=np.float32).reshape(grid.shape)
    )
    initial = truth.copy()
    with RecToolsIR(
        projection_geometry=scan, volume_geometry=grid, OS_number=subsets
    ) as rec:
        data = rec.forward(truth)
        original = data.copy()
        result = rec.OSEM(data, initial=initial, iterations=3, recon_mask_radius=None)
        cp.testing.assert_allclose(result, truth, rtol=2e-6, atol=2e-7)
        cp.testing.assert_allclose(rec.forward(result), data, rtol=2e-6, atol=2e-7)
        cp.testing.assert_array_equal(data, original)
        cp.testing.assert_array_equal(initial, truth)
