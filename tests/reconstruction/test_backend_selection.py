"""Backend dispatch, lazy resources, output buffers and geometry replacement."""

from dataclasses import replace
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from tomobar.projectors._usfft.geometry import USFFTGeometryBackend
from tomobar.geometry import (
    Parallel2DGeometry,
    Parallel3DGeometry,
    Volume2DGeometry,
    Volume3DGeometry,
)


def geometries(dim=3):
    if dim == 2:
        return Parallel2DGeometry(
            angles=[0, 1, 2], detector_shape=(8,)
        ), Volume2DGeometry((8, 8))
    return Parallel3DGeometry(
        angles=[0, 1, 2], detector_shape=(2, 8)
    ), Volume3DGeometry((2, 8, 8))


@pytest.mark.parametrize("dim", [2, 3])
def test_usfft_selection_is_lazy_and_closes(mock_reconstruction, dim):
    ProjectionOperator = mock_reconstruction.ProjectionOperator

    scan, grid = geometries(dim)
    with ProjectionOperator(
        projection_geometry=scan, volume_geometry=grid, backend="usfft", subset_count=2
    ) as operator:
        assert operator.backend == "usfft"
        assert isinstance(operator._backend, USFFTGeometryBackend)
        assert operator._backend._numerical is None
        assert operator.domain_shape == grid.shape
        assert operator.subset_shape(1)[-2] == 1
    assert operator._backend._closed
    assert not mock_reconstruction.backend.astra.create_projector.called


def test_iterative_selection_and_transactional_replacement(mock_reconstruction):
    scan, grid = geometries()
    with mock_reconstruction.iterative.RecToolsIR(
        projection_geometry=scan,
        volume_geometry=grid,
        backend="usfft",
        OS_number=2,
        detector_padding=1,
    ) as rec:
        assert rec.backend == "usfft"
        previous = rec.operator
        rec.replace_geometry(replace(scan, centre_of_rotation=0.5))
        assert previous._backend._closed
        assert rec.backend == "usfft"
        assert rec.operator.range_shape == (2, 3, 10)
        current = rec.operator
        with pytest.raises(ValueError, match="shifts"):
            rec.replace_geometry(replace(scan, detector_shifts=np.ones((3, 2))))
        assert rec.operator is current
        assert not current._backend._closed
        assert rec.operator.subset_count == 2
    assert not mock_reconstruction.backend.astra.create_projector.called


@pytest.mark.parametrize("backend", ["unknown", None, "USFFT"])
def test_invalid_selection(mock_reconstruction, backend):
    ProjectionOperator = mock_reconstruction.ProjectionOperator

    scan, grid = geometries()
    for cls in (ProjectionOperator, mock_reconstruction.iterative.RecToolsIR):
        with pytest.raises(ValueError, match="backend"):
            cls(projection_geometry=scan, volume_geometry=grid, backend=backend)


@pytest.mark.parametrize("case", ["rectangular", "rows", "width", "shifts", "vectors"])
def test_unsupported_geometry_rejected_without_cuda(case):
    scan, grid = geometries()
    if case == "rectangular":
        grid = Volume3DGeometry((2, 7, 8))
    elif case == "rows":
        grid = Volume3DGeometry((3, 8, 8))
    elif case == "width":
        scan = replace(scan, detector_shape=(2, 7))
    elif case == "shifts":
        scan = replace(scan, detector_shifts=np.ones((3, 2)))
    else:
        scan = Parallel3DGeometry.from_vectors(
            vectors=scan.to_vectors(), detector_shape=(2, 8)
        )
    with pytest.raises(ValueError, match="USFFT"):
        USFFTGeometryBackend(projection_geometry=scan, volume_geometry=grid)


@pytest.mark.parametrize("dim", [2, 3])
def test_adapter_output_contract_and_execution(monkeypatch, dim):
    scan, grid = geometries(dim)
    instances = []

    class Numerical:
        def __init__(self, **options):
            instances.append(self)
            self.options = options

        def forward(self, data, *, subset_index, out=None):
            assert data.ndim == 3
            count = 3 if subset_index is None else 1
            if out is not None:
                out.fill(1)
                return out
            return np.ones((data.shape[0], count, 8), dtype=np.float32)

        def backproject(self, data, *, subset_index, out=None):
            assert data.ndim == 3
            return np.ones((data.shape[0], 8, 8), dtype=np.float32)

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "cupy", np)
    monkeypatch.setitem(
        sys.modules,
        "tomobar.projectors._usfft.backend",
        SimpleNamespace(USFFTBackend=Numerical),
    )
    backend = USFFTGeometryBackend(
        projection_geometry=scan, volume_geometry=grid, subset_count=2
    )
    image = np.zeros(grid.shape, dtype=np.float32)
    shape = (*scan.projection_shape[:-2], 1, 8)
    output = np.full(shape, np.nan, dtype=np.float32)
    assert backend.forward(image, subset_index=1, out=output) is output
    assert np.all(output == 1)
    assert len(instances) == 1
    assert backend.backproject(output, subset_index=1).shape == grid.shape
    with pytest.raises(ValueError, match="float32"):
        backend.forward(image, subset_index=1, out=output.astype(np.float64))
    overlapping = image.ravel()[: np.prod(scan.projection_shape)].reshape(
        scan.projection_shape
    )
    with pytest.raises(ValueError, match="overlap"):
        backend.forward(image, out=overlapping)
    backend.close()
    assert instances[0].closed
    with pytest.raises(RuntimeError, match="closed"):
        backend.forward(image)


@pytest.mark.parametrize("dim", [2, 3])
def test_public_usfft_gpu(gpu, dim):
    from tomobar.projectors.operator import ProjectionOperator

    scan, grid = geometries(dim)
    with ProjectionOperator(
        projection_geometry=scan, volume_geometry=grid, backend="usfft", subset_count=2
    ) as operator:
        image = gpu.ones(grid.shape, dtype=gpu.float32)
        projections = operator.forward(image)
        assert projections.shape == scan.projection_shape
        assert bool(gpu.isfinite(projections).all())
        subset_shape = operator.subset_shape(1)
        subset = gpu.empty(subset_shape, dtype=gpu.float32)
        assert operator.forward(image, subset_index=1, out=subset) is subset
        gpu.testing.assert_allclose(
            subset, projections[..., 1::2, :], rtol=1e-5, atol=1e-6
        )
        output = gpu.empty(grid.shape, dtype=gpu.float32)
        assert operator.backproject(subset, subset_index=1, out=output) is output
        assert bool(gpu.isfinite(output).all())


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("offset", [-2, -0.375, 0, 0.5, 3])
def test_scalar_cor_supported_without_resources(dim, offset):
    scan, grid = geometries(dim)
    scan = replace(scan, centre_of_rotation=offset)
    with_backend = USFFTGeometryBackend(projection_geometry=scan, volume_geometry=grid)
    assert with_backend.projection_geometry.centre_of_rotation == offset
    assert with_backend._numerical is None
    with_backend.close()


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("constant", [False, True])
def test_all_nonzero_shifts_rejected(dim, constant):
    scan, grid = geometries(dim)
    shifts = np.ones_like(scan.detector_shifts)
    if not constant:
        shifts[0] = 0
    # A scalar CoR remains valid but cannot make detector shifts acceptable.
    scan = replace(scan, detector_shifts=shifts, centre_of_rotation=0.75)
    with pytest.raises(ValueError, match="backend='astra'"):
        USFFTGeometryBackend(projection_geometry=scan, volume_geometry=grid)


def test_usfft_alignment_fails_before_processing_data(mock_reconstruction):
    scan, grid = geometries()
    with mock_reconstruction.iterative.RecToolsIR(
        projection_geometry=scan, volume_geometry=grid, backend="usfft"
    ) as rec:
        with pytest.raises(ValueError, match="alignment requires backend='astra'"):
            rec.align(None)
        assert rec.geometry_version == 0
        assert rec.operator._backend._numerical is None


@pytest.mark.parametrize("method", ["SIRT", "OSEM"])
def test_ray_weight_solvers_reject_usfft_before_data_preparation(
    mock_reconstruction, method
):
    scan, grid = geometries()
    with mock_reconstruction.iterative.RecToolsIR(
        projection_geometry=scan, volume_geometry=grid, backend="usfft"
    ) as rec:
        for initialise in (
            lambda: rec.initialise_solver(method, None),
            lambda: getattr(rec, method)(None, iterations=1),
        ):
            with pytest.raises(ValueError, match="requires backend='astra'"):
                initialise()
        assert rec.operator._backend._numerical is None


@pytest.mark.parametrize("backend", ["astra", "usfft"])
def test_neutral_backend_ownership_and_replacement(mock_reconstruction, backend):
    scan, grid = geometries()
    with mock_reconstruction.iterative.RecToolsIR(
        projection_geometry=scan, volume_geometry=grid, backend=backend, OS_number=2
    ) as rec:
        original = rec._projection_backend
        assert original is rec.operator._backend
        assert rec.Atools is original  # Existing ASTRA callers retain their alias.
        assert "Atools" not in rec.__dict__
        assert original.backend_name == backend
        assert original.subset_count == 2
        if backend == "usfft":
            assert not hasattr(original, "ordsub_number")
        rec.replace_geometry(replace(scan, centre_of_rotation=0.25))
        assert original._closed
        assert rec._projection_backend is rec.operator._backend
        assert rec.Atools is rec._projection_backend
        assert rec.backend == backend
        assert rec._projection_backend.subset_count == 2


def test_usfft_geometry_replacement_discards_execution_caches(gpu):
    from tomobar.reconstruction.iterative import RecToolsIR

    scan, grid = geometries()
    with RecToolsIR(
        projection_geometry=scan, volume_geometry=grid, backend="usfft", OS_number=2
    ) as rec:
        image = gpu.ones(grid.shape, dtype=gpu.float32)
        rec.operator.forward(image, subset_index=1)
        previous = rec._projection_backend._numerical
        phase = previous._view_plans[1].phase.copy()
        rec.replace_geometry(replace(scan, centre_of_rotation=0.375))
        assert previous._closed
        assert previous._workspace is None
        assert previous._view_plans == {}
        assert rec._projection_backend._numerical is None
        rec.operator.forward(image, subset_index=1)
        current = rec._projection_backend._numerical
        assert current is not previous
        assert not bool(gpu.allclose(current._view_plans[1].phase, phase))
